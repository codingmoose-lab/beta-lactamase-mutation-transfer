from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GINEConv
from tqdm.auto import tqdm


class ResidueGraphNeuralEvaluationError(RuntimeError):
    pass


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_INDEX = {value: index for index, value in enumerate(AMINO_ACIDS)}
SUBSTITUTIONS = tuple(
    (wild_type, mutant)
    for wild_type in AMINO_ACIDS
    for mutant in AMINO_ACIDS
    if mutant != wild_type
)
SUBSTITUTION_INDEX = {value: index for index, value in enumerate(SUBSTITUTIONS)}
SECONDARY_STRUCTURE_STATES = ("helix", "strand", "coil")
SECONDARY_STRUCTURE_INDEX = {
    value: index for index, value in enumerate(SECONDARY_STRUCTURE_STATES)
}

CORE_SCHEME_IDS = (
    "ndm1_position_nested5",
    "vim2_2020_position_nested5",
    "ndm1_to_vim2_amp_source_cv5",
    "vim2_to_ndm1_amp_source_cv5",
    "vim2_2022_mutation_nested5",
)

CONDITION_MODE_BY_SCHEME = {
    "ndm1_position_nested5": "categorical_condition",
    "vim2_2020_position_nested5": "categorical_condition",
    "ndm1_to_vim2_amp_source_cv5": "none",
    "vim2_to_ndm1_amp_source_cv5": "none",
    "vim2_2022_mutation_nested5": "log2_concentration",
}

CENTRE_NUMERIC_COLUMNS = (
    "heavy_atom_count",
    "sasa_angstrom2",
    "relative_sasa_wilke",
    "ca_neighbour_count_8_angstrom",
    "ca_neighbour_count_10_angstrom",
    "ca_neighbour_count_12_angstrom",
    "heavy_atom_contact_count_4_5_angstrom",
)

EDGE_CONTINUOUS_COLUMNS = (
    "ca_distance_angstrom",
    "minimum_heavy_atom_distance_angstrom",
)

EDGE_BINARY_COLUMNS = (
    "peptide_adjacent",
    "heavy_atom_contact_4_5_angstrom",
)

TARGET_COLUMN = "score_analysis"
CENTRE_MODEL = "centre_residue_neural_regressor"
GRAPH_MODEL = "residue_graph_neural_regressor"
MODEL_IDS = (CENTRE_MODEL, GRAPH_MODEL)

REFERENCE_MODEL_IDS = (
    "experimental_structure_descriptor_ridge",
    "residue_neighbourhood_descriptor_ridge",
    "structure_matched_mutation_condition_ridge",
)

COMPARISON_KEY = (
    "scheme_id",
    "evaluation_iteration",
    "evaluation_role",
    "observation_id",
)

MATCHED_FIELDS = (
    "task_id",
    "evaluation_group_id",
    "protein_id",
    "assay_position",
    "wild_type",
    "mutant",
    "antibiotic",
    "concentration",
    "temperature_c",
    "y_true",
)

HIDDEN_WIDTH = 128
GRAPH_LAYER_COUNT = 2
DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 400
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 4096
GRADIENT_NORM_LIMIT = 5.0
TRAINING_REPEATS = 3
SELECTION_SEED = 32001
FINAL_REPEAT_SEEDS = (32101, 32102, 32103)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_residue_graph_neural_bootstrap"
TARGET_ABSOLUTE_TOLERANCE = 1e-12

EXPECTED_GRAPH_NODE_COUNT = 469
EXPECTED_GRAPH_EDGE_COUNT = 2416
EXPECTED_PREDICTIONS_PER_MODEL = 118204
EXPECTED_REFERENCE_PREDICTION_COUNT = 3 * EXPECTED_PREDICTIONS_PER_MODEL


class ContinuousTransform:
    def __init__(self, mean: np.ndarray, sd: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.sd = np.asarray(sd, dtype=np.float32)

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.sd


class ConditionTransform:
    def __init__(
        self,
        mode: str,
        categorical_columns: Sequence[str] = (),
        categories: Mapping[str, Sequence[str]] | None = None,
        concentration_mean: float = 0.0,
        concentration_sd: float = 1.0,
    ):
        self.mode = mode
        self.categorical_columns = tuple(categorical_columns)
        self.categories = {
            key: tuple(values) for key, values in (categories or {}).items()
        }
        self.concentration_mean = float(concentration_mean)
        self.concentration_sd = float(concentration_sd)

    @property
    def output_dimension(self) -> int:
        if self.mode == "none":
            return 0
        if self.mode == "log2_concentration":
            return 1
        return sum(len(self.categories[column]) for column in self.categorical_columns)

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mode == "none":
            return np.zeros((len(frame), 0), dtype=np.float32)

        if self.mode == "log2_concentration":
            concentration = _numeric_concentration(frame)
            values = np.log2(concentration)
            return ((values - self.concentration_mean) / self.concentration_sd).astype(
                np.float32
            )[:, None]

        blocks = []
        for column in self.categorical_columns:
            categories = self.categories[column]
            category_index = {value: index for index, value in enumerate(categories)}
            encoded = np.zeros((len(frame), len(categories)), dtype=np.float32)
            for row_index, value in enumerate(_category_values(frame[column])):
                if value not in category_index:
                    raise ResidueGraphNeuralEvaluationError(
                        f"Condition value absent from fitting rows: {column}={value}"
                    )
                encoded[row_index, category_index[value]] = 1.0
            blocks.append(encoded)
        return np.concatenate(blocks, axis=1)


class ObservationBatch:
    def __init__(
        self,
        target_node_index: torch.Tensor,
        substitution: torch.Tensor,
        condition: torch.Tensor,
        target: torch.Tensor,
    ):
        self.target_node_index = target_node_index
        self.substitution = substitution
        self.condition = condition
        self.target = target


class CentreResidueNeuralRegressor(nn.Module):
    def __init__(self, node_input_dim: int, condition_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.ReLU(),
        )
        readout_dim = HIDDEN_WIDTH + len(SUBSTITUTIONS) + condition_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        target_node_index: torch.Tensor,
        substitution: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, edge_features
        encoded_nodes = self.node_encoder(node_features)
        centre = encoded_nodes[target_node_index]
        combined = torch.cat((centre, substitution, condition), dim=1)
        return self.readout(combined).squeeze(1)


class ResidueGraphNeuralRegressor(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int, condition_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, HIDDEN_WIDTH),
            nn.ReLU(),
        )
        self.convolutions = nn.ModuleList()
        self.normalisations = nn.ModuleList()
        for _ in range(GRAPH_LAYER_COUNT):
            message_network = nn.Sequential(
                nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
                nn.ReLU(),
                nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            )
            self.convolutions.append(
                GINEConv(message_network, edge_dim=edge_input_dim, train_eps=True)
            )
            self.normalisations.append(nn.LayerNorm(HIDDEN_WIDTH))

        readout_dim = HIDDEN_WIDTH + len(SUBSTITUTIONS) + condition_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        target_node_index: torch.Tensor,
        substitution: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.node_encoder(node_features)
        for convolution, normalisation in zip(
            self.convolutions, self.normalisations, strict=True
        ):
            updated = convolution(hidden, edge_index, edge_features)
            hidden = normalisation(hidden + updated)
            hidden = torch.relu(hidden)
        centre = hidden[target_node_index]
        combined = torch.cat((centre, substitution, condition), dim=1)
        return self.readout(combined).squeeze(1)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _write_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalised_text(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".15g")
    return str(value)


def _category_values(series: pd.Series) -> list[str]:
    return [
        _normalised_text(value) if _normalised_text(value) else "<missing>"
        for value in series
    ]


def _numeric_concentration(frame: pd.DataFrame) -> np.ndarray:
    column_options = (
        "ampicillin_concentration_ug_per_ml",
        "concentration",
    )
    for column in column_options:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).all() and (values > 0.0).all():
                return values
    raise ResidueGraphNeuralEvaluationError(
        "Positive numerical concentration values are required for this analysis"
    )


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], table_name: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ResidueGraphNeuralEvaluationError(
            f"Missing columns in {table_name}: {missing}"
        )


def _fit_continuous_transform(values: np.ndarray) -> ContinuousTransform:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ResidueGraphNeuralEvaluationError("Continuous fitting array is empty")
    if not np.isfinite(values).all():
        raise ResidueGraphNeuralEvaluationError(
            "Continuous fitting array is not finite"
        )
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    return ContinuousTransform(mean.astype(np.float32), sd.astype(np.float32))


def _fit_condition_transform(frame: pd.DataFrame, mode: str) -> ConditionTransform:
    if mode == "none":
        return ConditionTransform("none")

    if mode == "log2_concentration":
        values = np.log2(_numeric_concentration(frame))
        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        if not math.isfinite(sd) or sd <= 0.0:
            sd = 1.0
        return ConditionTransform(
            "log2_concentration",
            concentration_mean=mean,
            concentration_sd=sd,
        )

    columns = tuple(
        column
        for column in ("antibiotic", "concentration", "temperature_c")
        if column in frame.columns
    )
    if not columns:
        raise ResidueGraphNeuralEvaluationError(
            "Categorical assay-condition fields are unavailable"
        )
    categories = {
        column: tuple(sorted(set(_category_values(frame[column]))))
        for column in columns
    }
    return ConditionTransform(
        "categorical_condition",
        categorical_columns=columns,
        categories=categories,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_deterministic_execution() -> None:
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _load_tables(repo_root: Path) -> Mapping[str, pd.DataFrame]:
    paths = {
        "observations": repo_root / "data/processed/missense_observations.csv",
        "residue_features": repo_root / "data/processed/residue_features.csv",
        "membership": repo_root
        / "data/interim/modelling_views/scheme_observation_membership.csv",
        "schemes": repo_root
        / "data/interim/evaluation_partitions/partition_schemes.csv",
        "assignments": repo_root
        / "data/interim/evaluation_partitions/group_assignments.csv",
        "nodes": repo_root / "data/interim/residue_graphs/residue_graph_nodes.csv",
        "edges": repo_root / "data/interim/residue_graphs/residue_graph_edges.csv",
        "reference_predictions": repo_root
        / "data/interim/residue_neighbourhood_evaluation"
        / "residue_neighbourhood_predictions.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ResidueGraphNeuralEvaluationError(
            f"Required input files are missing: {missing}"
        )
    return {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}


def _validate_input_contract(tables: Mapping[str, pd.DataFrame]) -> None:
    observations = tables["observations"]
    residue_features = tables["residue_features"]
    membership = tables["membership"]
    schemes = tables["schemes"]
    assignments = tables["assignments"]
    nodes = tables["nodes"]
    edges = tables["edges"]
    reference_predictions = tables["reference_predictions"]

    _require_columns(
        observations,
        (
            "observation_id",
            "protein_id",
            "assay_position",
            "wild_type",
            "mutant",
            TARGET_COLUMN,
        ),
        "missense observations",
    )
    _require_columns(
        residue_features,
        ("protein_id", "assay_position", *CENTRE_NUMERIC_COLUMNS),
        "residue features",
    )
    _require_columns(membership, ("scheme_id", "observation_id"), "scheme membership")
    _require_columns(
        schemes,
        ("scheme_id", "task_id", "group_type"),
        "partition schemes",
    )
    _require_columns(
        assignments,
        ("scheme_id", "iteration", "group_id", "split_role", "cv_fold"),
        "group assignments",
    )
    _require_columns(
        nodes,
        (
            "protein_id",
            "node_index",
            "assay_position",
            "assay_residue",
            "heavy_atom_count",
            "sasa_angstrom2",
            "relative_sasa_wilke",
            "secondary_structure_3state",
        ),
        "residue graph nodes",
    )
    _require_columns(
        edges,
        (
            "protein_id",
            "source_node_index",
            "target_node_index",
            *EDGE_CONTINUOUS_COLUMNS,
            "sequence_separation",
            *EDGE_BINARY_COLUMNS,
        ),
        "residue graph edges",
    )
    _require_columns(
        reference_predictions,
        (*COMPARISON_KEY, "model_id", *MATCHED_FIELDS, "y_pred"),
        "Script-31 predictions",
    )

    if set(CORE_SCHEME_IDS) - set(schemes["scheme_id"].astype(str)):
        raise ResidueGraphNeuralEvaluationError("One or more core analyses are absent")
    if len(nodes) != EXPECTED_GRAPH_NODE_COUNT:
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected residue graph node count: {len(nodes)}"
        )
    if len(edges) != EXPECTED_GRAPH_EDGE_COUNT:
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected residue graph edge count: {len(edges)}"
        )
    if len(reference_predictions) != EXPECTED_REFERENCE_PREDICTION_COUNT:
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected Script-31 prediction count: {len(reference_predictions)}"
        )
    counts = reference_predictions["model_id"].value_counts().to_dict()
    if set(counts) != set(REFERENCE_MODEL_IDS):
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected Script-31 model identifiers: {sorted(counts)}"
        )
    if any(
        int(counts[model_id]) != EXPECTED_PREDICTIONS_PER_MODEL for model_id in counts
    ):
        raise ResidueGraphNeuralEvaluationError(
            "Script-31 model populations are incomplete"
        )
    if reference_predictions.duplicated([*COMPARISON_KEY, "model_id"]).any():
        raise ResidueGraphNeuralEvaluationError(
            "Script-31 prediction keys are duplicated"
        )
    if not np.isfinite(
        pd.to_numeric(observations[TARGET_COLUMN], errors="coerce")
    ).all():
        raise ResidueGraphNeuralEvaluationError("Analysis outcomes are not finite")


def _build_analysis_observations(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    observations = tables["observations"].copy()
    residue_features = tables["residue_features"].copy()
    nodes = tables["nodes"].copy()

    residue_columns = ["protein_id", "assay_position", *CENTRE_NUMERIC_COLUMNS]
    residue_subset = residue_features[residue_columns].drop_duplicates(
        ["protein_id", "assay_position"]
    )
    if len(residue_subset) != len(
        residue_features.drop_duplicates(["protein_id", "assay_position"])
    ):
        raise ResidueGraphNeuralEvaluationError("Residue feature keys are ambiguous")

    node_subset = nodes[
        [
            "protein_id",
            "assay_position",
            "node_index",
            "secondary_structure_3state",
            "assay_residue",
        ]
    ].copy()
    merged = observations.merge(
        residue_subset,
        on=["protein_id", "assay_position"],
        how="left",
        validate="many_to_one",
    ).merge(
        node_subset,
        on=["protein_id", "assay_position"],
        how="left",
        validate="many_to_one",
    )
    return merged


def _scheme_population(
    analysis_observations: pd.DataFrame,
    membership: pd.DataFrame,
    scheme_id: str,
) -> pd.DataFrame:
    member_ids = membership.loc[
        membership["scheme_id"].astype(str).eq(scheme_id), "observation_id"
    ].astype(str)
    if member_ids.empty:
        raise ResidueGraphNeuralEvaluationError(f"No membership rows for {scheme_id}")
    member_ids = member_ids.drop_duplicates()
    population = analysis_observations[
        analysis_observations["observation_id"].astype(str).isin(set(member_ids))
    ].copy()
    if population.empty:
        raise ResidueGraphNeuralEvaluationError(f"No observation rows for {scheme_id}")

    required_structure_columns = [
        "node_index",
        "secondary_structure_3state",
        *CENTRE_NUMERIC_COLUMNS,
    ]
    missing_columns = [
        column
        for column in required_structure_columns
        if column not in population.columns
    ]
    if missing_columns:
        raise ResidueGraphNeuralEvaluationError(
            "Structure-supported population fields are missing: "
            + ", ".join(missing_columns)
        )

    supported = population["node_index"].notna()
    supported &= population["secondary_structure_3state"].notna()
    for column in CENTRE_NUMERIC_COLUMNS:
        supported &= pd.to_numeric(population[column], errors="coerce").notna()
    population = population[supported].copy()

    if population.empty:
        raise ResidueGraphNeuralEvaluationError(
            f"No structure-supported observation rows for {scheme_id}"
        )
    return population


def _group_column(scheme_row: pd.Series, observations: pd.DataFrame) -> str:
    group_type = str(scheme_row["group_type"])
    if group_type in observations.columns:
        return group_type
    if (
        group_type == "position_group_id"
        and "position_group_id" in observations.columns
    ):
        return "position_group_id"
    if (
        group_type == "mutation_group_id"
        and "mutation_group_id" in observations.columns
    ):
        return "mutation_group_id"
    raise ResidueGraphNeuralEvaluationError(
        f"Observation table does not contain group field {group_type}"
    )


def _rows_for_groups(
    population: pd.DataFrame,
    group_column: str,
    group_values: Sequence[object],
) -> pd.DataFrame:
    allowed = {_normalised_text(value) for value in group_values}
    observed = population[group_column].map(_normalised_text)
    return population[observed.isin(allowed)].copy()


def _reference_test_rows(
    reference_predictions: pd.DataFrame,
    scheme_id: str,
    evaluation_iteration: str,
) -> pd.DataFrame:
    reference = reference_predictions[
        reference_predictions["model_id"].eq("experimental_structure_descriptor_ridge")
        & reference_predictions["scheme_id"].astype(str).eq(scheme_id)
        & reference_predictions["evaluation_iteration"]
        .astype(str)
        .eq(str(evaluation_iteration))
    ].copy()
    if reference.empty:
        raise ResidueGraphNeuralEvaluationError(
            f"No Script-31 test rows for {scheme_id}/{evaluation_iteration}"
        )
    return reference


def _nested_contexts(
    tables: Mapping[str, pd.DataFrame],
    analysis_observations: pd.DataFrame,
    scheme_id: str,
) -> list[Mapping[str, object]]:
    schemes = tables["schemes"]
    assignments = tables["assignments"]
    membership = tables["membership"]
    reference_predictions = tables["reference_predictions"]

    scheme_rows = schemes[schemes["scheme_id"].astype(str).eq(scheme_id)]
    if len(scheme_rows) != 1:
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected scheme rows for {scheme_id}"
        )
    scheme_row = scheme_rows.iloc[0]
    population = _scheme_population(analysis_observations, membership, scheme_id)
    group_column = _group_column(scheme_row, population)
    scheme_assignments = assignments[
        assignments["scheme_id"].astype(str).eq(scheme_id)
    ].copy()

    contexts = []
    iterations = sorted(
        {_normalised_text(value) for value in scheme_assignments["iteration"]},
        key=lambda value: int(value) if value.isdigit() else value,
    )
    for iteration in iterations:
        current = scheme_assignments[
            scheme_assignments["iteration"].map(_normalised_text).eq(iteration)
        ]
        roles = set(current["split_role"].astype(str))
        if not {"train", "validation", "test"}.issubset(roles):
            continue
        train = _rows_for_groups(
            population,
            group_column,
            current.loc[current["split_role"].eq("train"), "group_id"].tolist(),
        )
        validation = _rows_for_groups(
            population,
            group_column,
            current.loc[current["split_role"].eq("validation"), "group_id"].tolist(),
        )
        test = _rows_for_groups(
            population,
            group_column,
            current.loc[current["split_role"].eq("test"), "group_id"].tolist(),
        )
        reference = _reference_test_rows(reference_predictions, scheme_id, iteration)
        if set(test["observation_id"].astype(str)) != set(
            reference["observation_id"].astype(str)
        ):
            raise ResidueGraphNeuralEvaluationError(
                f"Test observations disagree with Script-31 for {scheme_id}/{iteration}"
            )
        contexts.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme_row["task_id"]),
                "evaluation_iteration": iteration,
                "evaluation_role": "outer_test",
                "group_column": group_column,
                "train": train,
                "validation": validation,
                "test": test,
                "reference": reference,
            }
        )
    if len(contexts) != 5:
        raise ResidueGraphNeuralEvaluationError(
            f"Expected five outer iterations for {scheme_id}, observed {len(contexts)}"
        )
    return contexts


def _cross_protein_context(
    tables: Mapping[str, pd.DataFrame],
    analysis_observations: pd.DataFrame,
    scheme_id: str,
) -> Mapping[str, object]:
    schemes = tables["schemes"]
    assignments = tables["assignments"]
    membership = tables["membership"]
    reference_predictions = tables["reference_predictions"]

    scheme_rows = schemes[schemes["scheme_id"].astype(str).eq(scheme_id)]
    if len(scheme_rows) != 1:
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected scheme rows for {scheme_id}"
        )
    scheme_row = scheme_rows.iloc[0]
    population = _scheme_population(analysis_observations, membership, scheme_id)
    group_column = _group_column(scheme_row, population)
    current = assignments[assignments["scheme_id"].astype(str).eq(scheme_id)].copy()
    roles = set(current["split_role"].astype(str))
    if not {"source_model_selection", "target_test"}.issubset(roles):
        raise ResidueGraphNeuralEvaluationError(
            f"Cross-protein roles are incomplete for {scheme_id}: {sorted(roles)}"
        )

    source_assignment = current[
        current["split_role"].eq("source_model_selection")
    ].copy()
    target_assignment = current[current["split_role"].eq("target_test")].copy()
    source = _rows_for_groups(
        population, group_column, source_assignment["group_id"].tolist()
    )
    target = _rows_for_groups(
        population, group_column, target_assignment["group_id"].tolist()
    )

    if scheme_id == "ndm1_to_vim2_amp_source_cv5":
        source_protein, target_protein = "NDM-1", "VIM-2"
    elif scheme_id == "vim2_to_ndm1_amp_source_cv5":
        source_protein, target_protein = "VIM-2", "NDM-1"
    else:
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected cross-protein analysis identifier: {scheme_id}"
        )
    source = source[source["protein_id"].astype(str).eq(source_protein)].copy()
    target = target[target["protein_id"].astype(str).eq(target_protein)].copy()
    if source.empty or target.empty:
        raise ResidueGraphNeuralEvaluationError(
            f"Cross-protein source or target population is empty for {scheme_id}"
        )
    reference = _reference_test_rows(reference_predictions, scheme_id, "target")
    if set(target["observation_id"].astype(str)) != set(
        reference["observation_id"].astype(str)
    ):
        raise ResidueGraphNeuralEvaluationError(
            f"Target observations disagree with Script-31 for {scheme_id}"
        )

    folds = []
    fold_values = sorted(
        {
            int(value)
            for value in pd.to_numeric(source_assignment["cv_fold"], errors="coerce")
            if pd.notna(value)
        }
    )
    if fold_values != list(range(5)):
        raise ResidueGraphNeuralEvaluationError(
            f"Expected source CV folds 0-4 for {scheme_id}, observed {fold_values}"
        )
    source_group_text = source_assignment["group_id"].map(_normalised_text)
    for fold in fold_values:
        validation_groups = set(
            source_group_text[
                pd.to_numeric(source_assignment["cv_fold"], errors="coerce").eq(fold)
            ]
        )
        train_groups = set(source_group_text) - validation_groups
        folds.append(
            {
                "fold": fold,
                "train": _rows_for_groups(source, group_column, sorted(train_groups)),
                "validation": _rows_for_groups(
                    source, group_column, sorted(validation_groups)
                ),
            }
        )
    return {
        "scheme_id": scheme_id,
        "task_id": str(scheme_row["task_id"]),
        "evaluation_iteration": "target",
        "evaluation_role": "target_test",
        "group_column": group_column,
        "source": source,
        "target": target,
        "reference": reference,
        "source_folds": folds,
    }


def _make_contexts(
    tables: Mapping[str, pd.DataFrame], analysis_observations: pd.DataFrame
) -> list[Mapping[str, object]]:
    eligibility_column = "experimental_structure_task_eligible"
    if eligibility_column not in analysis_observations.columns:
        raise ResidueGraphNeuralEvaluationError(
            "Experimental-structure eligibility is absent from the observation table"
        )

    eligible = analysis_observations[
        analysis_observations[eligibility_column].fillna(False).astype(bool)
    ].copy()

    if eligible["observation_id"].nunique() != 109408:
        raise ResidueGraphNeuralEvaluationError(
            "Unexpected experimental-structure observation count: "
            f"{eligible['observation_id'].nunique()}"
        )

    contexts: list[Mapping[str, object]] = []
    for scheme_id in CORE_SCHEME_IDS:
        if "source_cv5" in scheme_id:
            contexts.append(_cross_protein_context(tables, eligible, scheme_id))
        else:
            contexts.extend(_nested_contexts(tables, eligible, scheme_id))

    if len(contexts) != 17:
        raise ResidueGraphNeuralEvaluationError(
            f"Expected 17 neural evaluation contexts, observed {len(contexts)}"
        )

    return contexts


def _node_lookup(nodes: pd.DataFrame) -> Mapping[tuple[str, int], int]:
    lookup: dict[tuple[str, int], int] = {}
    for global_index, row in nodes.reset_index(drop=True).iterrows():
        key = (str(row["protein_id"]), int(row["assay_position"]))
        if key in lookup:
            raise ResidueGraphNeuralEvaluationError(f"Duplicated graph node key: {key}")
        lookup[key] = int(global_index)
    return lookup


def _training_proteins(frame: pd.DataFrame) -> tuple[str, ...]:
    proteins = tuple(sorted(set(frame["protein_id"].astype(str))))
    if not proteins:
        raise ResidueGraphNeuralEvaluationError("Fitting rows contain no protein")
    return proteins


def _graph_tensors(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    residue_features: pd.DataFrame,
    fitting_rows: pd.DataFrame,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mapping[tuple[str, int], int]]:
    node_frame = nodes.copy().reset_index(drop=True)
    residue_subset = residue_features[
        ["protein_id", "assay_position", *CENTRE_NUMERIC_COLUMNS]
    ].drop_duplicates(["protein_id", "assay_position"])
    node_frame = node_frame.drop(
        columns=[
            column for column in CENTRE_NUMERIC_COLUMNS if column in node_frame.columns
        ],
        errors="ignore",
    ).merge(
        residue_subset,
        on=["protein_id", "assay_position"],
        how="left",
        validate="one_to_one",
    )
    if node_frame[list(CENTRE_NUMERIC_COLUMNS)].isna().any().any():
        raise ResidueGraphNeuralEvaluationError(
            "One or more graph nodes lack centre-residue numerical measurements"
        )

    fitting_proteins = _training_proteins(fitting_rows)
    node_numeric = node_frame[list(CENTRE_NUMERIC_COLUMNS)].to_numpy(dtype=float)
    fitting_node_mask = (
        node_frame["protein_id"].astype(str).isin(fitting_proteins).to_numpy()
    )
    node_transform = _fit_continuous_transform(node_numeric[fitting_node_mask])
    scaled_node_numeric = node_transform.apply(node_numeric)

    amino = np.zeros((len(node_frame), len(AMINO_ACIDS)), dtype=np.float32)
    secondary = np.zeros(
        (len(node_frame), len(SECONDARY_STRUCTURE_STATES)), dtype=np.float32
    )
    for row_index, row in node_frame.iterrows():
        residue = str(row["assay_residue"])
        state = str(row["secondary_structure_3state"])
        if residue not in AMINO_ACID_INDEX:
            raise ResidueGraphNeuralEvaluationError(
                f"Unexpected graph residue symbol: {residue}"
            )
        if state not in SECONDARY_STRUCTURE_INDEX:
            raise ResidueGraphNeuralEvaluationError(
                f"Unexpected secondary-structure state: {state}"
            )
        amino[row_index, AMINO_ACID_INDEX[residue]] = 1.0
        secondary[row_index, SECONDARY_STRUCTURE_INDEX[state]] = 1.0

    node_features = np.concatenate((amino, secondary, scaled_node_numeric), axis=1)

    per_protein_offset = {}
    for protein_id, subset in node_frame.groupby("protein_id", sort=False):
        local_indices = subset["node_index"].to_numpy(dtype=int)
        expected = np.arange(len(subset), dtype=int)
        if not np.array_equal(local_indices, expected):
            raise ResidueGraphNeuralEvaluationError(
                f"Node indices are not contiguous within {protein_id}"
            )
        per_protein_offset[str(protein_id)] = int(subset.index.min())

    directed_source = []
    directed_target = []
    edge_rows = []
    for _, row in edges.iterrows():
        protein_id = str(row["protein_id"])
        offset = per_protein_offset[protein_id]
        source = offset + int(row["source_node_index"])
        target = offset + int(row["target_node_index"])
        base = [
            float(row["ca_distance_angstrom"]),
            float(row["minimum_heavy_atom_distance_angstrom"]),
            math.log1p(float(row["sequence_separation"])),
            float(row["peptide_adjacent"]),
            float(row["heavy_atom_contact_4_5_angstrom"]),
        ]
        directed_source.extend((source, target))
        directed_target.extend((target, source))
        edge_rows.extend((base, base))

    edge_array = np.asarray(edge_rows, dtype=float)
    directed_protein = np.repeat(edges["protein_id"].astype(str).to_numpy(), 2)
    fitting_edge_mask = np.isin(directed_protein, np.asarray(fitting_proteins))
    continuous_transform = _fit_continuous_transform(edge_array[fitting_edge_mask, :3])
    edge_array[:, :3] = continuous_transform.apply(edge_array[:, :3])
    if not np.isfinite(edge_array).all():
        raise ResidueGraphNeuralEvaluationError("Graph edge inputs are not finite")

    edge_index = torch.tensor(
        [directed_source, directed_target], dtype=torch.long, device=device
    )
    return (
        torch.tensor(node_features, dtype=torch.float32, device=device),
        edge_index,
        torch.tensor(edge_array, dtype=torch.float32, device=device),
        _node_lookup(node_frame),
    )


def _substitution_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = np.zeros((len(frame), len(SUBSTITUTIONS)), dtype=np.float32)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        key = (str(row.wild_type), str(row.mutant))
        if key not in SUBSTITUTION_INDEX:
            raise ResidueGraphNeuralEvaluationError(f"Unexpected substitution: {key}")
        values[row_index, SUBSTITUTION_INDEX[key]] = 1.0
    return values


def _node_indices(
    frame: pd.DataFrame, lookup: Mapping[tuple[str, int], int]
) -> np.ndarray:
    output = np.empty(len(frame), dtype=np.int64)
    for index, row in enumerate(frame.itertuples(index=False)):
        key = (str(row.protein_id), int(row.assay_position))
        if key not in lookup:
            raise ResidueGraphNeuralEvaluationError(
                f"Observation lacks graph node: {key}"
            )
        output[index] = int(lookup[key])
    return output


def _prepare_observation_arrays(
    frame: pd.DataFrame,
    node_lookup: Mapping[tuple[str, int], int],
    condition_transform: ConditionTransform,
    target_mean: float,
    target_sd: float,
) -> Mapping[str, np.ndarray]:
    targets = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ResidueGraphNeuralEvaluationError("Observation outcomes are not finite")
    return {
        "node_index": _node_indices(frame, node_lookup),
        "substitution": _substitution_matrix(frame),
        "condition": condition_transform.encode(frame),
        "target": ((targets - target_mean) / target_sd).astype(np.float32),
        "target_original": targets.astype(np.float64),
    }


def _target_transform(frame: pd.DataFrame) -> tuple[float, float]:
    values = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ResidueGraphNeuralEvaluationError("Fitting outcomes are unavailable")
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    if not math.isfinite(sd) or sd <= 0.0:
        sd = 1.0
    return mean, sd


def _model(
    model_id: str,
    node_input_dim: int,
    edge_input_dim: int,
    condition_dim: int,
    device: torch.device,
) -> nn.Module:
    if model_id == CENTRE_MODEL:
        return CentreResidueNeuralRegressor(node_input_dim, condition_dim).to(device)
    if model_id == GRAPH_MODEL:
        return ResidueGraphNeuralRegressor(
            node_input_dim, edge_input_dim, condition_dim
        ).to(device)
    raise ResidueGraphNeuralEvaluationError(f"Unknown neural model: {model_id}")


def _loader(
    arrays: Mapping[str, np.ndarray], batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays["node_index"]).long(),
        torch.from_numpy(arrays["substitution"]).float(),
        torch.from_numpy(arrays["condition"]).float(),
        torch.from_numpy(arrays["target"]).float(),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _predict_standardised(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = _loader(arrays, batch_size, shuffle=False)
    result = []
    with torch.no_grad():
        for node_index, substitution, condition, _target in loader:
            prediction = model(
                node_features,
                edge_index,
                edge_features,
                node_index.to(device, non_blocking=True),
                substitution.to(device, non_blocking=True),
                condition.to(device, non_blocking=True),
            )
            result.append(prediction.detach().cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def _fit_with_validation(
    model_id: str,
    fitting_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    residue_features: pd.DataFrame,
    condition_mode: str,
    device: torch.device,
    seed: int,
    description: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[int, float, list[Mapping[str, object]]]:
    _set_seed(seed)
    condition_transform = _fit_condition_transform(fitting_rows, condition_mode)
    target_mean, target_sd = _target_transform(fitting_rows)
    node_features, edge_index, edge_features, node_lookup = _graph_tensors(
        nodes, edges, residue_features, fitting_rows, device
    )
    fitting_arrays = _prepare_observation_arrays(
        fitting_rows, node_lookup, condition_transform, target_mean, target_sd
    )
    validation_arrays = _prepare_observation_arrays(
        validation_rows, node_lookup, condition_transform, target_mean, target_sd
    )
    model = _model(
        model_id,
        node_features.shape[1],
        edge_features.shape[1],
        condition_transform.output_dimension,
        device,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.MSELoss()
    training_loader = _loader(fitting_arrays, batch_size, shuffle=True)

    best_epoch = 0
    best_validation_mae = float("inf")
    epochs_without_improvement = 0
    history = []
    progress = tqdm(
        range(1, max_epochs + 1),
        desc=description,
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in progress:
        model.train()
        total_squared_loss = 0.0
        total_rows = 0
        for node_index, substitution, condition, target in training_loader:
            optimiser.zero_grad(set_to_none=True)
            prediction = model(
                node_features,
                edge_index,
                edge_features,
                node_index.to(device, non_blocking=True),
                substitution.to(device, non_blocking=True),
                condition.to(device, non_blocking=True),
            )
            target_device = target.to(device, non_blocking=True)
            loss = loss_function(prediction, target_device)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRADIENT_NORM_LIMIT)
            optimiser.step()
            total_squared_loss += float(loss.detach().cpu()) * len(target)
            total_rows += len(target)

        training_loss = total_squared_loss / max(1, total_rows)
        validation_standardised = _predict_standardised(
            model,
            validation_arrays,
            node_features,
            edge_index,
            edge_features,
            device,
            batch_size,
        )
        validation_prediction = validation_standardised * target_sd + target_mean
        validation_mae = float(
            np.mean(
                np.abs(validation_prediction - validation_arrays["target_original"])
            )
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_mae": validation_mae,
            }
        )

        if validation_mae < best_validation_mae - 1e-12:
            best_validation_mae = validation_mae
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        progress.set_postfix(
            train_loss=f"{training_loss:.4f}",
            val_mae=f"{validation_mae:.4f}",
            best=f"{best_validation_mae:.4f}",
            refresh=False,
        )
        if epochs_without_improvement >= patience:
            break

    if best_epoch <= 0 or not math.isfinite(best_validation_mae):
        raise ResidueGraphNeuralEvaluationError("Neural epoch selection failed")
    return best_epoch, best_validation_mae, history


def _fit_fixed_epochs_and_predict(
    model_id: str,
    fitting_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    residue_features: pd.DataFrame,
    condition_mode: str,
    device: torch.device,
    seed: int,
    epochs: int,
    description: str,
    batch_size: int,
) -> tuple[np.ndarray, list[Mapping[str, object]]]:
    _set_seed(seed)
    condition_transform = _fit_condition_transform(fitting_rows, condition_mode)
    target_mean, target_sd = _target_transform(fitting_rows)
    node_features, edge_index, edge_features, node_lookup = _graph_tensors(
        nodes, edges, residue_features, fitting_rows, device
    )
    fitting_arrays = _prepare_observation_arrays(
        fitting_rows, node_lookup, condition_transform, target_mean, target_sd
    )
    test_arrays = _prepare_observation_arrays(
        test_rows, node_lookup, condition_transform, target_mean, target_sd
    )
    model = _model(
        model_id,
        node_features.shape[1],
        edge_features.shape[1],
        condition_transform.output_dimension,
        device,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.MSELoss()
    training_loader = _loader(fitting_arrays, batch_size, shuffle=True)
    history = []
    progress = tqdm(
        range(1, epochs + 1),
        desc=description,
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in progress:
        model.train()
        total_squared_loss = 0.0
        total_rows = 0
        for node_index, substitution, condition, target in training_loader:
            optimiser.zero_grad(set_to_none=True)
            prediction = model(
                node_features,
                edge_index,
                edge_features,
                node_index.to(device, non_blocking=True),
                substitution.to(device, non_blocking=True),
                condition.to(device, non_blocking=True),
            )
            target_device = target.to(device, non_blocking=True)
            loss = loss_function(prediction, target_device)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRADIENT_NORM_LIMIT)
            optimiser.step()
            total_squared_loss += float(loss.detach().cpu()) * len(target)
            total_rows += len(target)
        training_loss = total_squared_loss / max(1, total_rows)
        history.append({"epoch": epoch, "training_loss": training_loss})
        progress.set_postfix(train_loss=f"{training_loss:.4f}", refresh=False)

    standardised = _predict_standardised(
        model,
        test_arrays,
        node_features,
        edge_index,
        edge_features,
        device,
        batch_size,
    )
    prediction = standardised * target_sd + target_mean
    if not np.isfinite(prediction).all():
        raise ResidueGraphNeuralEvaluationError("Neural predictions are not finite")
    return prediction, history


def _selection_for_context(
    context: Mapping[str, object],
    model_id: str,
    tables: Mapping[str, pd.DataFrame],
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[int, list[Mapping[str, object]]]:
    nodes = tables["nodes"]
    edges = tables["edges"]
    residue_features = tables["residue_features"]
    scheme_id = str(context["scheme_id"])
    condition_mode = CONDITION_MODE_BY_SCHEME[scheme_id]
    history_rows: list[Mapping[str, object]] = []

    if "source_folds" not in context:
        epoch_count, validation_mae, history = _fit_with_validation(
            model_id,
            context["train"],
            context["validation"],
            nodes,
            edges,
            residue_features,
            condition_mode,
            device,
            SELECTION_SEED,
            f"{scheme_id} {context['evaluation_iteration']} {model_id} selection",
            max_epochs,
            patience,
            batch_size,
        )
        for row in history:
            history_rows.append(
                {
                    "scheme_id": scheme_id,
                    "evaluation_iteration": context["evaluation_iteration"],
                    "model_id": model_id,
                    "fit_role": "validation_epoch_selection",
                    "source_cv_fold": "",
                    **row,
                }
            )
        history_rows.append(
            {
                "scheme_id": scheme_id,
                "evaluation_iteration": context["evaluation_iteration"],
                "model_id": model_id,
                "fit_role": "selected_epoch",
                "source_cv_fold": "",
                "epoch": epoch_count,
                "training_loss": np.nan,
                "validation_mae": validation_mae,
            }
        )
        return epoch_count, history_rows

    selected_epochs = []
    fold_maes = []
    for fold_record in context["source_folds"]:
        fold = int(fold_record["fold"])
        epoch_count, validation_mae, history = _fit_with_validation(
            model_id,
            fold_record["train"],
            fold_record["validation"],
            nodes,
            edges,
            residue_features,
            condition_mode,
            device,
            SELECTION_SEED + fold,
            f"{scheme_id} source-CV {fold} {model_id}",
            max_epochs,
            patience,
            batch_size,
        )
        selected_epochs.append(epoch_count)
        fold_maes.append(validation_mae)
        for row in history:
            history_rows.append(
                {
                    "scheme_id": scheme_id,
                    "evaluation_iteration": "target",
                    "model_id": model_id,
                    "fit_role": "source_cv_epoch_selection",
                    "source_cv_fold": fold,
                    **row,
                }
            )

    selected_epoch = int(round(float(np.median(selected_epochs))))
    selected_epoch = max(1, selected_epoch)
    history_rows.append(
        {
            "scheme_id": scheme_id,
            "evaluation_iteration": "target",
            "model_id": model_id,
            "fit_role": "selected_epoch",
            "source_cv_fold": "",
            "epoch": selected_epoch,
            "training_loss": np.nan,
            "validation_mae": float(np.mean(fold_maes)),
        }
    )
    return selected_epoch, history_rows


def _prediction_records(
    reference: pd.DataFrame,
    model_id: str,
    repeat_predictions: Sequence[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = reference.reset_index(drop=True)
    repeat_frames = []
    for repeat_index, prediction in enumerate(repeat_predictions, start=1):
        if len(prediction) != len(reference):
            raise ResidueGraphNeuralEvaluationError(
                "Neural test prediction length mismatch"
            )
        frame = reference[[*COMPARISON_KEY, *MATCHED_FIELDS]].copy()
        frame["model_id"] = model_id
        frame["training_repeat"] = repeat_index
        frame["y_pred"] = prediction
        repeat_frames.append(frame)
    repeat_frame = pd.concat(repeat_frames, ignore_index=True)
    averaged = reference[[*COMPARISON_KEY, *MATCHED_FIELDS]].copy()
    averaged["model_id"] = model_id
    averaged["y_pred"] = np.mean(np.vstack(repeat_predictions), axis=0)
    return repeat_frame, averaged


def _fit_context(
    context: Mapping[str, object],
    model_id: str,
    tables: Mapping[str, pd.DataFrame],
    device: torch.device,
    repeats: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_epoch, selection_history = _selection_for_context(
        context, model_id, tables, device, max_epochs, patience, batch_size
    )
    scheme_id = str(context["scheme_id"])
    condition_mode = CONDITION_MODE_BY_SCHEME[scheme_id]
    if "source_folds" in context:
        fitting_rows = context["source"].copy()
        test_rows = context["target"].copy()
    else:
        fitting_rows = pd.concat(
            [context["train"], context["validation"]], ignore_index=True
        ).drop_duplicates("observation_id")
        test_rows = context["test"].copy()

    reference = context["reference"].copy()
    ordered_test = reference[["observation_id"]].merge(
        test_rows,
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    if ordered_test[TARGET_COLUMN].isna().any():
        raise ResidueGraphNeuralEvaluationError("Unable to order neural test rows")

    repeat_predictions = []
    final_history = []
    seeds = list(FINAL_REPEAT_SEEDS)
    if repeats > len(seeds):
        seeds.extend(
            FINAL_REPEAT_SEEDS[-1] + offset + 1
            for offset in range(repeats - len(seeds))
        )
    for repeat_index in range(repeats):
        prediction, history = _fit_fixed_epochs_and_predict(
            model_id,
            fitting_rows,
            ordered_test,
            tables["nodes"],
            tables["edges"],
            tables["residue_features"],
            condition_mode,
            device,
            int(seeds[repeat_index]),
            selected_epoch,
            (
                f"{scheme_id} {context['evaluation_iteration']} {model_id} "
                f"repeat {repeat_index + 1}/{repeats}"
            ),
            batch_size,
        )
        repeat_predictions.append(prediction)
        for row in history:
            final_history.append(
                {
                    "scheme_id": scheme_id,
                    "evaluation_iteration": context["evaluation_iteration"],
                    "model_id": model_id,
                    "fit_role": "final_refit",
                    "source_cv_fold": "",
                    "training_repeat": repeat_index + 1,
                    "selected_epoch_count": selected_epoch,
                    **row,
                }
            )

    repeat_frame, averaged = _prediction_records(
        reference, model_id, repeat_predictions
    )
    history_frame = pd.DataFrame([*selection_history, *final_history])
    history_frame["training_repeat"] = history_frame.get("training_repeat", "")
    history_frame["selected_epoch_count"] = history_frame.get(
        "selected_epoch_count", selected_epoch
    )
    return repeat_frame, averaged, history_frame


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    if len(y_true) < 2:
        return {"spearman": np.nan, "pearson": np.nan, "mae": np.nan, "rmse": np.nan}
    spearman = float(spearmanr(y_true, y_pred).statistic)
    pearson = float(pearsonr(y_true, y_pred).statistic)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"spearman": spearman, "pearson": pearson, "mae": mae, "rmse": rmse}


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme_id, model_id, iteration), subset in predictions.groupby(
        ["scheme_id", "model_id", "evaluation_iteration"], sort=True
    ):
        values = _metric_values(
            subset["y_true"].to_numpy(dtype=float),
            subset["y_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "scheme_id": scheme_id,
                "model_id": model_id,
                "evaluation_iteration": iteration,
                "n_observations": len(subset),
                "n_groups": subset["evaluation_group_id"].nunique(),
                **values,
            }
        )
    for (scheme_id, model_id), subset in predictions.groupby(
        ["scheme_id", "model_id"], sort=True
    ):
        values = _metric_values(
            subset["y_true"].to_numpy(dtype=float),
            subset["y_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "scheme_id": scheme_id,
                "model_id": model_id,
                "evaluation_iteration": "pooled",
                "n_observations": len(subset),
                "n_groups": subset["evaluation_group_id"].nunique(),
                **values,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["scheme_id", "model_id", "evaluation_iteration"], kind="stable")
        .reset_index(drop=True)
    )


def _bootstrap_seed(scheme_id: str, model_a: str, model_b: str, metric: str) -> int:
    text = "|".join((BOOTSTRAP_SEED, scheme_id, model_a, model_b, metric))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _metric_difference(
    metric: str, a: Mapping[str, float], b: Mapping[str, float]
) -> float:
    if metric in {"spearman", "pearson"}:
        return float(a[metric] - b[metric])
    return float(b[metric] - a[metric])


def _paired_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    comparison_pairs = (
        (GRAPH_MODEL, CENTRE_MODEL),
        (GRAPH_MODEL, "residue_neighbourhood_descriptor_ridge"),
        (GRAPH_MODEL, "experimental_structure_descriptor_ridge"),
    )
    rows = []
    for scheme_id in CORE_SCHEME_IDS:
        scheme = predictions[predictions["scheme_id"].astype(str).eq(scheme_id)]
        for model_a, model_b in comparison_pairs:
            a = scheme[scheme["model_id"].eq(model_a)].copy()
            b = scheme[scheme["model_id"].eq(model_b)].copy()
            paired = a[
                [*COMPARISON_KEY, "evaluation_group_id", "y_true", "y_pred"]
            ].merge(
                b[[*COMPARISON_KEY, "y_true", "y_pred"]],
                on=list(COMPARISON_KEY),
                how="inner",
                validate="one_to_one",
                suffixes=("_a", "_b"),
            )
            if len(paired) != len(a) or len(paired) != len(b):
                raise ResidueGraphNeuralEvaluationError(
                    f"Paired population mismatch for {scheme_id}/{model_a}/{model_b}"
                )
            if not np.allclose(
                paired["y_true_a"].to_numpy(dtype=float),
                paired["y_true_b"].to_numpy(dtype=float),
                rtol=0.0,
                atol=TARGET_ABSOLUTE_TOLERANCE,
            ):
                raise ResidueGraphNeuralEvaluationError(
                    f"Paired outcomes disagree for {scheme_id}/{model_a}/{model_b}"
                )
            y_true = paired["y_true_a"].to_numpy(dtype=float)
            pred_a = paired["y_pred_a"].to_numpy(dtype=float)
            pred_b = paired["y_pred_b"].to_numpy(dtype=float)
            metrics_a = _metric_values(y_true, pred_a)
            metrics_b = _metric_values(y_true, pred_b)
            group_ids = sorted(paired["evaluation_group_id"].astype(str).unique())
            groups = {
                group_id: paired[paired["evaluation_group_id"].astype(str).eq(group_id)]
                for group_id in group_ids
            }
            for metric in ("spearman", "pearson", "mae", "rmse"):
                observed = _metric_difference(metric, metrics_a, metrics_b)
                rng = np.random.default_rng(
                    _bootstrap_seed(scheme_id, model_a, model_b, metric)
                )
                bootstrap_values = []
                for _ in range(BOOTSTRAP_REPLICATES):
                    sampled_ids = rng.choice(
                        group_ids, size=len(group_ids), replace=True
                    )
                    sampled = pd.concat(
                        [groups[group_id] for group_id in sampled_ids],
                        ignore_index=True,
                    )
                    truth = sampled["y_true_a"].to_numpy(dtype=float)
                    values_a = _metric_values(
                        truth, sampled["y_pred_a"].to_numpy(dtype=float)
                    )
                    values_b = _metric_values(
                        truth, sampled["y_pred_b"].to_numpy(dtype=float)
                    )
                    value = _metric_difference(metric, values_a, values_b)
                    if math.isfinite(value):
                        bootstrap_values.append(value)
                if not bootstrap_values:
                    raise ResidueGraphNeuralEvaluationError(
                        f"No valid bootstrap replicates for {scheme_id}/{metric}"
                    )
                lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "model_a_value": metrics_a[metric],
                        "model_b_value": metrics_b[metric],
                        "paired_difference": observed,
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_valid_replicates": len(bootstrap_values),
                        "resampling_unit": "evaluation_group_id",
                        "difference_interpretation": "positive_favours_model_a",
                    }
                )
    return pd.DataFrame(rows)


def _parameter_count(
    model_id: str, node_dim: int, edge_dim: int, condition_dim: int
) -> int:
    model = _model(model_id, node_dim, edge_dim, condition_dim, torch.device("cpu"))
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _validate_new_predictions(
    reference_predictions: pd.DataFrame, neural_predictions: pd.DataFrame
) -> None:
    if len(neural_predictions) != 2 * EXPECTED_PREDICTIONS_PER_MODEL:
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected neural prediction count: {len(neural_predictions)}"
        )
    counts = neural_predictions["model_id"].value_counts().to_dict()
    if counts != {model_id: EXPECTED_PREDICTIONS_PER_MODEL for model_id in MODEL_IDS}:
        raise ResidueGraphNeuralEvaluationError(
            f"Unexpected neural model counts: {counts}"
        )
    if neural_predictions.duplicated([*COMPARISON_KEY, "model_id"]).any():
        raise ResidueGraphNeuralEvaluationError("Neural prediction keys are duplicated")
    reference = reference_predictions[
        reference_predictions["model_id"].eq("experimental_structure_descriptor_ridge")
    ][[*COMPARISON_KEY, *MATCHED_FIELDS]].copy()
    for model_id in MODEL_IDS:
        new = neural_predictions[neural_predictions["model_id"].eq(model_id)].copy()
        merged = reference.merge(
            new[[*COMPARISON_KEY, *MATCHED_FIELDS]],
            on=list(COMPARISON_KEY),
            how="outer",
            validate="one_to_one",
            suffixes=("_reference", "_new"),
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ResidueGraphNeuralEvaluationError(
                f"Neural and Script-31 test populations disagree for {model_id}"
            )
        for field in MATCHED_FIELDS:
            left = merged[f"{field}_reference"]
            right = merged[f"{field}_new"]
            if field == "y_true":
                same = np.allclose(
                    pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
                    rtol=0.0,
                    atol=TARGET_ABSOLUTE_TOLERANCE,
                    equal_nan=True,
                )
            elif pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
                right
            ):
                same = np.allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            else:
                same = (
                    left.fillna("")
                    .astype(str)
                    .reset_index(drop=True)
                    .equals(right.fillna("").astype(str).reset_index(drop=True))
                )
            if not bool(same):
                raise ResidueGraphNeuralEvaluationError(
                    f"Matched field disagrees with Script-31 for {model_id}: {field}"
                )


def inspect_residue_graph_neural_inputs(repo_root: Path) -> Mapping[str, object]:
    repo_root = Path(repo_root).resolve()
    tables = _load_tables(repo_root)
    _validate_input_contract(tables)
    analysis_observations = _build_analysis_observations(tables)
    contexts = _make_contexts(tables, analysis_observations)

    context_summary = []
    for context in contexts:
        if "source_folds" in context:
            context_summary.append(
                {
                    "scheme_id": context["scheme_id"],
                    "evaluation_iteration": "target",
                    "fitting_row_count": len(context["source"]),
                    "test_row_count": len(context["target"]),
                    "source_cv_fold_count": len(context["source_folds"]),
                }
            )
        else:
            context_summary.append(
                {
                    "scheme_id": context["scheme_id"],
                    "evaluation_iteration": context["evaluation_iteration"],
                    "training_row_count": len(context["train"]),
                    "validation_row_count": len(context["validation"]),
                    "test_row_count": len(context["test"]),
                }
            )

    return {
        "status": "PASS",
        "analysis_scope": "residue_graph_neural_mutation_effect_evaluation",
        "core_scheme_ids": list(CORE_SCHEME_IDS),
        "model_ids": list(MODEL_IDS),
        "graph_node_count": len(tables["nodes"]),
        "graph_edge_count": len(tables["edges"]),
        "evaluation_context_count": len(contexts),
        "reference_prediction_count": len(tables["reference_predictions"]),
        "contexts": context_summary,
        "cuda_available": torch.cuda.is_available(),
    }


def prepare_residue_graph_neural_evaluation(
    repo_root: Path,
    *,
    device_name: str = "cuda",
    repeats: int = TRAINING_REPEATS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, object],
]:
    repo_root = Path(repo_root).resolve()
    if repeats < 1:
        raise ResidueGraphNeuralEvaluationError(
            "Training repeat count must be positive"
        )
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ResidueGraphNeuralEvaluationError("Training controls must be positive")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ResidueGraphNeuralEvaluationError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    tables = _load_tables(repo_root)
    _validate_input_contract(tables)
    analysis_observations = _build_analysis_observations(tables)
    contexts = _make_contexts(tables, analysis_observations)

    repeat_frames = []
    averaged_frames = []
    history_frames = []
    overall = tqdm(
        total=len(contexts) * len(MODEL_IDS),
        desc="residue-graph neural analyses",
        unit="fit",
        dynamic_ncols=True,
        leave=True,
    )
    for context in contexts:
        for model_id in MODEL_IDS:
            overall.set_postfix(
                scheme=str(context["scheme_id"]),
                iteration=str(context["evaluation_iteration"]),
                model=model_id,
                refresh=False,
            )
            repeat_frame, averaged, history = _fit_context(
                context,
                model_id,
                tables,
                device,
                repeats,
                max_epochs,
                patience,
                batch_size,
            )
            repeat_frames.append(repeat_frame)
            averaged_frames.append(averaged)
            history_frames.append(history)
            overall.update(1)
    overall.close()

    repeat_predictions = pd.concat(repeat_frames, ignore_index=True)
    neural_predictions = pd.concat(averaged_frames, ignore_index=True)
    training_history = pd.concat(history_frames, ignore_index=True)
    _validate_new_predictions(tables["reference_predictions"], neural_predictions)

    reference_subset = tables["reference_predictions"][
        tables["reference_predictions"]["model_id"].isin(
            (
                "experimental_structure_descriptor_ridge",
                "residue_neighbourhood_descriptor_ridge",
                "structure_matched_mutation_condition_ridge",
            )
        )
    ].copy()
    combined_predictions = pd.concat(
        [reference_subset, neural_predictions], ignore_index=True, sort=False
    )
    metrics = _metrics(combined_predictions)
    paired = _paired_comparisons(combined_predictions)

    output_dir = repo_root / "data/interim/residue_graph_neural_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_path = output_dir / "residue_graph_neural_repeat_predictions.csv"
    prediction_path = output_dir / "residue_graph_neural_predictions.csv"
    history_path = output_dir / "residue_graph_neural_training_history.csv"
    metrics_path = output_dir / "residue_graph_neural_metrics.csv"
    paired_path = output_dir / "residue_graph_neural_paired_comparisons.csv"
    summary_path = output_dir / "residue_graph_neural_summary.json"

    _write_csv(repeat_predictions, repeat_path)
    _write_csv(neural_predictions, prediction_path)
    _write_csv(training_history, history_path)
    _write_csv(metrics, metrics_path)
    _write_csv(paired, paired_path)

    representative_condition_dim = _fit_condition_transform(
        analysis_observations.iloc[:1], "none"
    ).output_dimension
    node_dim = (
        len(AMINO_ACIDS) + len(SECONDARY_STRUCTURE_STATES) + len(CENTRE_NUMERIC_COLUMNS)
    )
    edge_dim = 5
    summary: dict[str, object] = {
        "status": "PASS",
        "analysis_scope": "residue_graph_neural_mutation_effect_evaluation",
        "model_ids": list(MODEL_IDS),
        "core_scheme_ids": list(CORE_SCHEME_IDS),
        "coordinate_source": "selected_experimental_pdb",
        "graph_node_count": len(tables["nodes"]),
        "graph_edge_count": len(tables["edges"]),
        "node_input_dimension": node_dim,
        "edge_input_dimension": edge_dim,
        "centre_numeric_inputs": list(CENTRE_NUMERIC_COLUMNS),
        "node_amino_acid_encoding": "20_residue_one_hot",
        "secondary_structure_states": list(SECONDARY_STRUCTURE_STATES),
        "substitution_encoding": "380_directed_missense_one_hot",
        "condition_modes": dict(CONDITION_MODE_BY_SCHEME),
        "graph_model_operator": "GINEConv",
        "graph_layer_count": GRAPH_LAYER_COUNT,
        "hidden_width": HIDDEN_WIDTH,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "gradient_norm_limit": GRADIENT_NORM_LIMIT,
        "training_repeats": repeats,
        "selection_seed": SELECTION_SEED,
        "final_repeat_seeds": list(FINAL_REPEAT_SEEDS[:repeats]),
        "loss": "mean_squared_error_on_fitting_standardised_outcome",
        "epoch_selection_measure": "validation_mae",
        "cross_protein_epoch_selection": "five_source_protein_grouped_cv_folds",
        "cross_protein_test_labels_used_for_epoch_selection": False,
        "outer_test_labels_used_for_epoch_selection": False,
        "target_outcome_used_as_model_input": False,
        "protein_id_used_as_direct_model_input": False,
        "assay_position_used_as_direct_model_input": False,
        "absolute_coordinates_used_as_direct_model_inputs": False,
        "mechanistic_reference_inputs_used": False,
        "reported_homologue_pair_inputs_used": False,
        "graph_message_passing_control": CENTRE_MODEL,
        "primary_graph_model": GRAPH_MODEL,
        "reference_models_reused_without_refitting": list(REFERENCE_MODEL_IDS),
        "repeat_prediction_count": len(repeat_predictions),
        "averaged_neural_prediction_count": len(neural_predictions),
        "prediction_count_by_neural_model": {
            key: int(value)
            for key, value in neural_predictions["model_id"]
            .value_counts()
            .to_dict()
            .items()
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_resampling_unit": "evaluation_group_id",
        "bootstrap_interval_method": "percentile",
        "bootstrap_quantiles": [0.025, 0.975],
        "difference_interpretation": "positive_favours_model_a",
        "software": {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_geometric": __import__("torch_geometric").__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "",
        },
        "input_sha256": {
            "missense_observations": _sha256(
                repo_root / "data/processed/missense_observations.csv"
            ),
            "residue_features": _sha256(
                repo_root / "data/processed/residue_features.csv"
            ),
            "scheme_observation_membership": _sha256(
                repo_root
                / "data/interim/modelling_views/scheme_observation_membership.csv"
            ),
            "partition_schemes": _sha256(
                repo_root / "data/interim/evaluation_partitions/partition_schemes.csv"
            ),
            "group_assignments": _sha256(
                repo_root / "data/interim/evaluation_partitions/group_assignments.csv"
            ),
            "residue_graph_nodes": _sha256(
                repo_root / "data/interim/residue_graphs/residue_graph_nodes.csv"
            ),
            "residue_graph_edges": _sha256(
                repo_root / "data/interim/residue_graphs/residue_graph_edges.csv"
            ),
            "script_31_predictions": _sha256(
                repo_root
                / "data/interim/residue_neighbourhood_evaluation"
                / "residue_neighbourhood_predictions.csv"
            ),
        },
        "outputs": [
            repeat_path.name,
            prediction_path.name,
            history_path.name,
            metrics_path.name,
            paired_path.name,
            summary_path.name,
        ],
    }
    del representative_condition_dim
    _write_json(summary, summary_path)
    return (
        repeat_predictions,
        neural_predictions,
        training_history,
        metrics,
        paired,
        summary,
    )


def run_residue_graph_neural_evaluation(
    repo_root: Path,
    *,
    device_name: str = "cuda",
    repeats: int = TRAINING_REPEATS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
) -> Mapping[str, object]:
    _configure_deterministic_execution()
    return prepare_residue_graph_neural_evaluation(
        repo_root,
        device_name=device_name,
        repeats=repeats,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )[-1]
