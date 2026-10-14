from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

import betalactamase_transfer.residue_graph_neural_evaluation as graph_eval


class ResidueGeometryNeuralEvaluationError(RuntimeError):
    pass


MODEL_ID = "egnn_residue_regressor"
STAGE32_GRAPH_MODEL_ID = "residue_graph_neural_regressor"
STAGE32_CENTRE_MODEL_ID = "centre_residue_neural_regressor"
REFERENCE_MODEL_IDS = (
    "experimental_structure_descriptor_ridge",
    "residue_neighbourhood_descriptor_ridge",
    "structure_matched_mutation_condition_ridge",
)
COMPARATOR_MODEL_IDS = (
    STAGE32_GRAPH_MODEL_ID,
    STAGE32_CENTRE_MODEL_ID,
    *REFERENCE_MODEL_IDS,
)
CORE_SCHEME_IDS = tuple(graph_eval.CORE_SCHEME_IDS)
TARGET_COLUMN = graph_eval.TARGET_COLUMN

HIDDEN_WIDTH = 128
GEOMETRY_LAYER_COUNT = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_NORM_LIMIT = 5.0
MAX_EPOCHS = 400
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 4096
TRAINING_REPEATS = 3
SELECTION_SEED = 33001
FINAL_REPEAT_SEEDS = (33101, 33102, 33103)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_residue_geometry_bootstrap"
TARGET_ABSOLUTE_TOLERANCE = 1e-12
INVARIANCE_ABSOLUTE_TOLERANCE = 2e-5
CHECKPOINT_DIRECTORY_NAME = "context_checkpoints"
CHECKPOINT_RECORD_NAME = "checkpoint_run.json"
TRAINING_COMPLETE_RECORD_NAME = "training_complete.json"

EXPECTED_STRUCTURE_OBSERVATION_COUNT = 109408
EXPECTED_PREDICTIONS_PER_MODEL = 118204
EXPECTED_CONTEXT_COUNT = 17
EXPECTED_GRAPH_NODE_COUNT = 469
EXPECTED_GRAPH_EDGE_COUNT = 2416

STAGE32_OUTPUT_HASHES = {
    "residue_graph_neural_metrics.csv": (
        "663792cd6ed3c4d414ebb63b0eaa5dc838a28faa2be47749a05b71016e8bb547"
    ),
    "residue_graph_neural_paired_comparisons.csv": (
        "a14385011647ce58de6cb73fdc3189cdddec1006b5d56e7060713088bfd12994"
    ),
    "residue_graph_neural_predictions.csv": (
        "98aa268232824720aecd2a8510d4169e4faba4f4dfd71c31659b92088af0203a"
    ),
    "residue_graph_neural_repeat_predictions.csv": (
        "a8464a9cb4f3153a6d190b63a4fa62bf96370facf2cd7b561b15e4963c8d5367"
    ),
    "residue_graph_neural_summary.json": (
        "4f32e32ebe85dc49a4b2486bfb734cd395d51788e58e1e9aac0521ea46ba6faa"
    ),
    "residue_graph_neural_training_history.csv": (
        "7b5ca9d0bb055636c21a44dc6576e94b951ab9876ab54d733d81b34025ed60e4"
    ),
}


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


def _context_token(context: Mapping[str, object]) -> str:
    return f"{context['scheme_id']}__{context['evaluation_iteration']}".replace(
        "/", "_"
    ).replace(" ", "_")


def _checkpoint_contract(
    device: torch.device,
    repeats: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> Mapping[str, object]:
    return {
        "model_id": MODEL_ID,
        "repeats": repeats,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "selection_seed": SELECTION_SEED,
        "final_repeat_seeds": list(FINAL_REPEAT_SEEDS[:repeats]),
        "geometry_eval_source_sha256": _sha256(Path(__file__)),
        "graph_eval_output_sha256": dict(STAGE32_OUTPUT_HASHES),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else ""
        ),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    }


def _prepare_checkpoint_directory(
    output_dir: Path,
    contract: Mapping[str, object],
) -> Path:
    checkpoint_dir = output_dir / CHECKPOINT_DIRECTORY_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    contract_path = checkpoint_dir / CHECKPOINT_RECORD_NAME
    if contract_path.exists():
        recorded = json.loads(contract_path.read_text(encoding="utf-8"))
        if recorded != dict(contract):
            raise ResidueGeometryNeuralEvaluationError(
                "Existing residue-geometry neural analysis checkpoints were produced "
                "under a different execution contract"
            )
    else:
        _write_json(contract, contract_path)
    return checkpoint_dir


def _context_checkpoint_paths(
    checkpoint_dir: Path,
    context: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    token = _context_token(context)
    return (
        checkpoint_dir / f"{token}__repeat_predictions.csv",
        checkpoint_dir / f"{token}__training_history.csv",
        checkpoint_dir / f"{token}__complete.json",
    )


def _write_context_checkpoint(
    checkpoint_dir: Path,
    context: Mapping[str, object],
    repeat_rows: list[Mapping[str, object]],
    history_rows: list[Mapping[str, object]],
    selected_epoch: int,
    test_rows: pd.DataFrame,
    repeats: int,
) -> None:
    repeat_path, history_path, complete_path = _context_checkpoint_paths(
        checkpoint_dir, context
    )
    repeat_frame = pd.DataFrame(repeat_rows)
    history_frame = pd.DataFrame(history_rows)
    expected_repeat_rows = len(test_rows) * repeats
    if len(repeat_frame) != expected_repeat_rows:
        raise ResidueGeometryNeuralEvaluationError(
            "Context checkpoint has an unexpected repeat-prediction count"
        )
    identity = [
        "scheme_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
        "model_id",
        "training_repeat",
    ]
    if repeat_frame.duplicated(identity).any():
        raise ResidueGeometryNeuralEvaluationError(
            "Context checkpoint contains duplicate repeat-prediction identities"
        )
    if set(repeat_frame["training_repeat"].astype(int)) != set(range(1, repeats + 1)):
        raise ResidueGeometryNeuralEvaluationError(
            "Context checkpoint does not contain the expected repeat numbers"
        )
    _write_csv(repeat_frame, repeat_path)
    _write_csv(history_frame, history_path)
    _write_json(
        {
            "scheme_id": str(context["scheme_id"]),
            "evaluation_iteration": str(context["evaluation_iteration"]),
            "model_id": MODEL_ID,
            "selected_epoch_count": int(selected_epoch),
            "test_row_count": len(test_rows),
            "training_repeats": repeats,
            "repeat_prediction_count": len(repeat_frame),
            "repeat_predictions_sha256": _sha256(repeat_path),
            "training_history_sha256": _sha256(history_path),
        },
        complete_path,
    )


def _load_context_checkpoint(
    checkpoint_dir: Path,
    context: Mapping[str, object],
    test_rows: pd.DataFrame,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int] | None:
    repeat_path, history_path, complete_path = _context_checkpoint_paths(
        checkpoint_dir, context
    )
    if not complete_path.exists():
        return None
    if not repeat_path.exists() or not history_path.exists():
        raise ResidueGeometryNeuralEvaluationError(
            "Completed context checkpoint is missing a data file"
        )
    record = json.loads(complete_path.read_text(encoding="utf-8"))
    expected = {
        "scheme_id": str(context["scheme_id"]),
        "evaluation_iteration": str(context["evaluation_iteration"]),
        "model_id": MODEL_ID,
        "test_row_count": len(test_rows),
        "training_repeats": repeats,
        "repeat_prediction_count": len(test_rows) * repeats,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ResidueGeometryNeuralEvaluationError(
                f"Context checkpoint contract mismatch: {key}"
            )
    if record.get("repeat_predictions_sha256") != _sha256(repeat_path):
        raise ResidueGeometryNeuralEvaluationError(
            "Context repeat-prediction checkpoint hash does not match"
        )
    if record.get("training_history_sha256") != _sha256(history_path):
        raise ResidueGeometryNeuralEvaluationError(
            "Context training-history checkpoint hash does not match"
        )
    repeat_frame = pd.read_csv(
        repeat_path, dtype={"evaluation_iteration": str, "observation_id": str}
    )
    history_frame = pd.read_csv(history_path, dtype={"evaluation_iteration": str})
    identity = [
        "scheme_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
        "model_id",
        "training_repeat",
    ]
    if repeat_frame.duplicated(identity).any():
        raise ResidueGeometryNeuralEvaluationError(
            "Loaded context checkpoint contains duplicate identities"
        )
    if len(repeat_frame) != len(test_rows) * repeats:
        raise ResidueGeometryNeuralEvaluationError(
            "Loaded context checkpoint has an unexpected row count"
        )
    expected_targets = test_rows[["observation_id", TARGET_COLUMN]].copy()
    expected_targets["observation_id"] = expected_targets["observation_id"].astype(str)
    saved_targets = (
        repeat_frame[["observation_id", "y_true"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    merged = expected_targets.merge(
        saved_targets, on="observation_id", how="outer", indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ResidueGeometryNeuralEvaluationError(
            "Context checkpoint observations do not match the current evaluation rows"
        )
    if not np.allclose(
        merged[TARGET_COLUMN].to_numpy(dtype=float),
        merged["y_true"].to_numpy(dtype=float),
        rtol=0.0,
        atol=TARGET_ABSOLUTE_TOLERANCE,
    ):
        raise ResidueGeometryNeuralEvaluationError(
            "Context checkpoint outcomes do not match the current evaluation rows"
        )
    return repeat_frame, history_frame, int(record["selected_epoch_count"])


def _configure_reproducible_execution() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size, values.shape[1]))
    output.index_add_(0, index, values)
    return output


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    total = _scatter_sum(values, index, size)
    counts = values.new_zeros((size, 1))
    counts.index_add_(0, index, values.new_ones((len(index), 1)))
    return total / counts.clamp_min(1.0)


class EGNNResidueLayer(nn.Module):
    """One E(n)-equivariant residue-message layer."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        message_input_dim = hidden_dim * 2 + edge_dim + 1
        self.message = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coordinate_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalisation = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        receiver = edge_index[0]
        sender = edge_index[1]
        relative = coordinates[receiver] - coordinates[sender]
        squared_distance = (relative * relative).sum(dim=1, keepdim=True)
        message = self.message(
            torch.cat(
                (
                    hidden[receiver],
                    hidden[sender],
                    edge_features,
                    squared_distance,
                ),
                dim=1,
            )
        )

        coordinate_factor = self.coordinate_weight(message)
        coordinate_change = _scatter_mean(
            relative * coordinate_factor,
            receiver,
            len(hidden),
        )
        coordinates = coordinates + coordinate_change

        aggregated = _scatter_mean(message, receiver, len(hidden))
        updated = self.node_update(torch.cat((hidden, aggregated), dim=1))
        hidden = torch.relu(self.normalisation(hidden + updated))
        return hidden, coordinates


class EGNNResidueRegressor(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int, condition_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, HIDDEN_WIDTH),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [
                EGNNResidueLayer(HIDDEN_WIDTH, edge_input_dim)
                for _ in range(GEOMETRY_LAYER_COUNT)
            ]
        )
        readout_dim = HIDDEN_WIDTH + len(graph_eval.SUBSTITUTIONS) + condition_dim
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
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        target_node_index: torch.Tensor,
        substitution: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.node_encoder(node_features)
        working_coordinates = coordinates
        for layer in self.layers:
            hidden, working_coordinates = layer(
                hidden,
                working_coordinates,
                edge_index,
                edge_features,
            )
        centre = hidden[target_node_index]
        combined = torch.cat((centre, substitution, condition), dim=1)
        return self.readout(combined).squeeze(1)


def _required_graph_eval_symbols() -> tuple[str, ...]:
    return (
        "_load_tables",
        "_build_analysis_observations",
        "_make_contexts",
        "_graph_tensors",
        "_fit_condition_transform",
        "_target_transform",
        "_prepare_observation_arrays",
        "_loader",
        "_metric_values",
        "_metric_difference",
        "CONDITION_MODE_BY_SCHEME",
        "SUBSTITUTIONS",
        "TARGET_COLUMN",
    )


def _verify_graph_eval_module() -> None:
    missing = [
        name for name in _required_graph_eval_symbols() if not hasattr(graph_eval, name)
    ]
    if missing:
        raise ResidueGeometryNeuralEvaluationError(
            "residue-graph neural analysis module does not provide required "
            "functions: " + ", ".join(missing)
        )


def _verify_graph_eval_outputs(repo_root: Path) -> None:
    directory = repo_root / "data/interim/residue_graph_neural_evaluation"
    for filename, expected_hash in STAGE32_OUTPUT_HASHES.items():
        path = directory / filename
        if not path.is_file():
            raise ResidueGeometryNeuralEvaluationError(
                f"Required residue-graph neural analysis result is absent: {path}"
            )
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise ResidueGeometryNeuralEvaluationError(
                f"Residue-graph neural analysis result identity changed: {filename}"
            )


def _coordinate_tensor(
    nodes: pd.DataFrame,
    node_lookup: Mapping[object, int],
    device: torch.device,
) -> torch.Tensor:
    coordinate_columns = ("ca_x", "ca_y", "ca_z")
    if not set(coordinate_columns).issubset(nodes.columns):
        raise ResidueGeometryNeuralEvaluationError(
            "C-alpha coordinate columns are absent from the graph-node table"
        )
    if "protein_id" not in nodes.columns:
        raise ResidueGeometryNeuralEvaluationError(
            "protein_id is absent from graph nodes"
        )

    lookup_keys = set()
    for key in node_lookup:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ResidueGeometryNeuralEvaluationError(
                "Residue-graph neural analysis node lookup does not use "
                "protein/position keys"
            )
        lookup_keys.add((str(key[0]), int(key[1])))

    candidate_columns = (
        "assay_position",
        "residue_position",
        "position",
        "reference_position",
    )
    matching_columns = []
    for position_column in candidate_columns:
        if position_column not in nodes.columns:
            continue
        key_frame = nodes[["protein_id", position_column]].copy()
        if key_frame.duplicated().any():
            continue
        node_keys = {
            (str(protein_id), int(position))
            for protein_id, position in key_frame.itertuples(index=False, name=None)
        }
        if node_keys == lookup_keys:
            matching_columns.append(position_column)

    if len(matching_columns) != 1:
        raise ResidueGeometryNeuralEvaluationError(
            "Could not match one graph-node position column to the residue-graph "
            "neural analysis node lookup"
        )
    position_column = matching_columns[0]

    frame = nodes[["protein_id", position_column, *coordinate_columns]].copy()
    by_key = {
        (str(row.protein_id), int(getattr(row, position_column))): np.asarray(
            [row.ca_x, row.ca_y, row.ca_z], dtype=np.float32
        )
        for row in frame.itertuples(index=False)
    }
    coordinates = np.full((len(node_lookup), 3), np.nan, dtype=np.float32)

    for key, global_index in node_lookup.items():
        normalised_key = (str(key[0]), int(key[1]))
        coordinates[int(global_index)] = by_key[normalised_key]

    if not np.isfinite(coordinates).all():
        raise ResidueGeometryNeuralEvaluationError("Graph coordinates are not finite")
    return torch.from_numpy(coordinates).to(device)


def _model(
    node_dim: int,
    edge_dim: int,
    condition_dim: int,
    device: torch.device,
) -> EGNNResidueRegressor:
    return EGNNResidueRegressor(node_dim, edge_dim, condition_dim).to(device)


def _predict_standardised(
    model: EGNNResidueRegressor,
    arrays: Mapping[str, np.ndarray],
    node_features: torch.Tensor,
    coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = graph_eval._loader(arrays, batch_size, shuffle=False)
    result = []
    model.eval()
    with torch.no_grad():
        for node_index, substitution, condition, _target in loader:
            prediction = model(
                node_features,
                coordinates,
                edge_index,
                edge_features,
                node_index.to(device, non_blocking=True),
                substitution.to(device, non_blocking=True),
                condition.to(device, non_blocking=True),
            )
            result.append(prediction.detach().cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def _fit_with_validation(
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
    condition_transform = graph_eval._fit_condition_transform(
        fitting_rows, condition_mode
    )
    target_mean, target_sd = graph_eval._target_transform(fitting_rows)
    node_features, edge_index, edge_features, node_lookup = graph_eval._graph_tensors(
        nodes, edges, residue_features, fitting_rows, device
    )
    coordinates = _coordinate_tensor(nodes, node_lookup, device)
    fitting_arrays = graph_eval._prepare_observation_arrays(
        fitting_rows, node_lookup, condition_transform, target_mean, target_sd
    )
    validation_arrays = graph_eval._prepare_observation_arrays(
        validation_rows, node_lookup, condition_transform, target_mean, target_sd
    )
    model = _model(
        node_features.shape[1],
        edge_features.shape[1],
        condition_transform.output_dimension,
        device,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.MSELoss()
    training_loader = graph_eval._loader(fitting_arrays, batch_size, shuffle=True)

    best_epoch = 0
    best_validation_mae = float("inf")
    epochs_without_improvement = 0
    history: list[Mapping[str, object]] = []
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
                coordinates,
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
            coordinates,
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
        raise ResidueGeometryNeuralEvaluationError("Epoch selection failed")
    return best_epoch, best_validation_mae, history


def _fit_fixed_epochs_and_predict(
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
    condition_transform = graph_eval._fit_condition_transform(
        fitting_rows, condition_mode
    )
    target_mean, target_sd = graph_eval._target_transform(fitting_rows)
    node_features, edge_index, edge_features, node_lookup = graph_eval._graph_tensors(
        nodes, edges, residue_features, fitting_rows, device
    )
    coordinates = _coordinate_tensor(nodes, node_lookup, device)
    fitting_arrays = graph_eval._prepare_observation_arrays(
        fitting_rows, node_lookup, condition_transform, target_mean, target_sd
    )
    test_arrays = graph_eval._prepare_observation_arrays(
        test_rows, node_lookup, condition_transform, target_mean, target_sd
    )
    model = _model(
        node_features.shape[1],
        edge_features.shape[1],
        condition_transform.output_dimension,
        device,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.MSELoss()
    training_loader = graph_eval._loader(fitting_arrays, batch_size, shuffle=True)

    history: list[Mapping[str, object]] = []
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
                coordinates,
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

    prediction_standardised = _predict_standardised(
        model,
        test_arrays,
        node_features,
        coordinates,
        edge_index,
        edge_features,
        device,
        batch_size,
    )
    return prediction_standardised * target_sd + target_mean, history


def _selection_for_context(
    context: Mapping[str, object],
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
    condition_mode = graph_eval.CONDITION_MODE_BY_SCHEME[scheme_id]
    history_rows: list[Mapping[str, object]] = []

    if "source_folds" not in context:
        epoch_count, validation_mae, history = _fit_with_validation(
            context["train"],
            context["validation"],
            nodes,
            edges,
            residue_features,
            condition_mode,
            device,
            SELECTION_SEED,
            f"{scheme_id} {context['evaluation_iteration']} {MODEL_ID} selection",
            max_epochs,
            patience,
            batch_size,
        )
        for row in history:
            history_rows.append(
                {
                    "scheme_id": scheme_id,
                    "evaluation_iteration": str(context["evaluation_iteration"]),
                    "model_id": MODEL_ID,
                    "fit_role": "validation_epoch_selection",
                    "source_cv_fold": "",
                    **row,
                }
            )
        history_rows.append(
            {
                "scheme_id": scheme_id,
                "evaluation_iteration": str(context["evaluation_iteration"]),
                "model_id": MODEL_ID,
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
            fold_record["train"],
            fold_record["validation"],
            nodes,
            edges,
            residue_features,
            condition_mode,
            device,
            SELECTION_SEED + fold,
            f"{scheme_id} source-CV {fold} {MODEL_ID}",
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
                    "model_id": MODEL_ID,
                    "fit_role": "source_cv_epoch_selection",
                    "source_cv_fold": fold,
                    **row,
                }
            )

    selected_epoch = max(1, int(round(float(np.median(selected_epochs)))))
    history_rows.append(
        {
            "scheme_id": scheme_id,
            "evaluation_iteration": "target",
            "model_id": MODEL_ID,
            "fit_role": "selected_epoch",
            "source_cv_fold": "",
            "epoch": selected_epoch,
            "training_loss": np.nan,
            "validation_mae": float(np.mean(fold_maes)),
        }
    )
    return selected_epoch, history_rows


def _test_rows(context: Mapping[str, object]) -> pd.DataFrame:
    if "target" in context:
        return context["target"]
    return context["test"]


def _fitting_rows(context: Mapping[str, object]) -> pd.DataFrame:
    if "fitting" in context:
        return context["fitting"]

    if "train" in context and "validation" in context:
        return pd.concat(
            (context["train"], context["validation"]),
            ignore_index=True,
        )

    if "source_folds" in context:
        source_populations = []

        for fold_record in context["source_folds"]:
            if "train" not in fold_record or "validation" not in fold_record:
                raise ResidueGeometryNeuralEvaluationError(
                    "Source-protein CV fold lacks fitting or validation rows"
                )

            fold_population = pd.concat(
                (
                    fold_record["train"],
                    fold_record["validation"],
                ),
                ignore_index=True,
            )

            if fold_population["observation_id"].duplicated().any():
                raise ResidueGeometryNeuralEvaluationError(
                    "Source-protein CV fold contains duplicate observations"
                )

            source_populations.append(fold_population)

        if not source_populations:
            raise ResidueGeometryNeuralEvaluationError(
                "Source-protein CV folds are absent"
            )

        reference_ids = set(source_populations[0]["observation_id"].astype(str))

        for fold_population in source_populations[1:]:
            current_ids = set(fold_population["observation_id"].astype(str))
            if current_ids != reference_ids:
                raise ResidueGeometryNeuralEvaluationError(
                    "Source-protein CV folds do not reconstruct "
                    "the same fitting population"
                )

        return source_populations[0]

    raise ResidueGeometryNeuralEvaluationError("Fitting rows are absent from context")


def _metadata_values(
    frame: pd.DataFrame, column: str, default: object = ""
) -> list[object]:
    if column in frame.columns:
        return frame[column].tolist()
    return [default] * len(frame)


def _prediction_records(
    context: Mapping[str, object],
    test_rows: pd.DataFrame,
    predictions: np.ndarray,
    training_repeat: int,
) -> list[Mapping[str, object]]:
    group_column = str(context["group_column"])
    if group_column not in test_rows.columns:
        raise ResidueGeometryNeuralEvaluationError(
            f"Evaluation-group column missing: {group_column}"
        )
    required = (
        "observation_id",
        "protein_id",
        "assay_position",
        "wild_type",
        "mutant",
        TARGET_COLUMN,
    )
    missing = [column for column in required if column not in test_rows.columns]
    if missing:
        raise ResidueGeometryNeuralEvaluationError(
            "Prediction metadata columns missing: " + ", ".join(missing)
        )

    antibiotic = _metadata_values(test_rows, "antibiotic")
    concentration = _metadata_values(test_rows, "concentration", np.nan)
    temperature = _metadata_values(test_rows, "temperature_c", np.nan)
    if all(pd.isna(value) for value in temperature):
        temperature = _metadata_values(test_rows, "temperature", np.nan)

    rows = []
    for index, row in enumerate(test_rows.itertuples(index=False)):
        values = row._asdict()
        rows.append(
            {
                "scheme_id": str(context["scheme_id"]),
                "evaluation_iteration": str(context["evaluation_iteration"]),
                "evaluation_role": str(context["evaluation_role"]),
                "observation_id": str(values["observation_id"]),
                "task_id": str(context["task_id"]),
                "evaluation_group_id": str(values[group_column]),
                "protein_id": str(values["protein_id"]),
                "assay_position": int(values["assay_position"]),
                "wild_type": str(values["wild_type"]),
                "mutant": str(values["mutant"]),
                "antibiotic": antibiotic[index],
                "concentration": concentration[index],
                "temperature_c": temperature[index],
                "y_true": float(values[TARGET_COLUMN]),
                "model_id": MODEL_ID,
                "training_repeat": int(training_repeat),
                "y_pred": float(predictions[index]),
            }
        )
    return rows


def _average_repeat_predictions(
    repeat_predictions: pd.DataFrame, expected_repeats: int = TRAINING_REPEATS
) -> pd.DataFrame:
    identity = [
        "scheme_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
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
        "model_id",
    ]
    counts = repeat_predictions.groupby(identity, dropna=False)[
        "training_repeat"
    ].nunique()
    if (
        len(counts) != EXPECTED_PREDICTIONS_PER_MODEL
        or not counts.eq(expected_repeats).all()
    ):
        raise ResidueGeometryNeuralEvaluationError(
            "Each prediction must contain all training repeats"
        )
    averaged = (
        repeat_predictions.groupby(identity, dropna=False, as_index=False)["y_pred"]
        .mean()
        .sort_values(identity, kind="mergesort")
        .reset_index(drop=True)
    )
    return averaged


def _load_comparator_predictions(repo_root: Path) -> pd.DataFrame:
    graph_eval_path = (
        repo_root
        / "data/interim/residue_graph_neural_evaluation"
        / "residue_graph_neural_predictions.csv"
    )
    neural = pd.read_csv(graph_eval_path, dtype={"evaluation_iteration": str})
    neural = neural[
        neural["model_id"].isin((STAGE32_GRAPH_MODEL_ID, STAGE32_CENTRE_MODEL_ID))
    ].copy()

    tables = graph_eval._load_tables(repo_root)
    reference = tables["reference_predictions"].copy()
    reference["evaluation_iteration"] = reference["evaluation_iteration"].astype(str)
    reference = reference[reference["model_id"].isin(REFERENCE_MODEL_IDS)].copy()

    required = list(neural.columns)
    missing = [column for column in required if column not in reference.columns]
    if missing:
        raise ResidueGeometryNeuralEvaluationError(
            "Residue-neighbourhood analysis comparator fields missing: "
            + ", ".join(missing)
        )
    reference = reference[required]
    combined = pd.concat((neural, reference), ignore_index=True)
    expected_rows = EXPECTED_PREDICTIONS_PER_MODEL * len(COMPARATOR_MODEL_IDS)
    if len(combined) != expected_rows:
        raise ResidueGeometryNeuralEvaluationError(
            f"Unexpected comparator prediction count: {len(combined)}"
        )
    return combined


def _metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme_id, model_id, iteration), subset in predictions.groupby(
        ["scheme_id", "model_id", "evaluation_iteration"], sort=True
    ):
        values = graph_eval._metric_values(
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
        values = graph_eval._metric_values(
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
    return pd.DataFrame(rows)


def _bootstrap_seed(scheme_id: str, model_b: str, metric: str) -> int:
    text = f"{BOOTSTRAP_SEED}|{scheme_id}|{MODEL_ID}|{model_b}|{metric}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _paired_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    progress = tqdm(
        total=len(CORE_SCHEME_IDS) * len(COMPARATOR_MODEL_IDS) * 4,
        desc="residue-geometry neural analysis grouped bootstrap",
        unit="comparison",
        dynamic_ncols=True,
    )
    key = [
        "scheme_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
    ]
    for scheme_id in CORE_SCHEME_IDS:
        scheme = predictions[predictions["scheme_id"].astype(str).eq(scheme_id)].copy()
        primary = scheme[scheme["model_id"].eq(MODEL_ID)].copy()
        for comparator in COMPARATOR_MODEL_IDS:
            other = scheme[scheme["model_id"].eq(comparator)].copy()
            paired = primary.merge(
                other, on=key, suffixes=("_a", "_b"), how="outer", indicator=True
            )
            if not paired["_merge"].eq("both").all():
                raise ResidueGeometryNeuralEvaluationError(
                    f"Paired population mismatch for {scheme_id}/{comparator}"
                )
            if not np.allclose(
                paired["y_true_a"].to_numpy(dtype=float),
                paired["y_true_b"].to_numpy(dtype=float),
                rtol=0.0,
                atol=TARGET_ABSOLUTE_TOLERANCE,
            ):
                raise ResidueGeometryNeuralEvaluationError(
                    f"Paired outcomes disagree for {scheme_id}/{comparator}"
                )
            y_true = paired["y_true_a"].to_numpy(dtype=float)
            pred_a = paired["y_pred_a"].to_numpy(dtype=float)
            pred_b = paired["y_pred_b"].to_numpy(dtype=float)
            metrics_a = graph_eval._metric_values(y_true, pred_a)
            metrics_b = graph_eval._metric_values(y_true, pred_b)
            group_ids = sorted(paired["evaluation_group_id_a"].astype(str).unique())
            groups = {
                group_id: paired[
                    paired["evaluation_group_id_a"].astype(str).eq(group_id)
                ]
                for group_id in group_ids
            }
            for metric in ("spearman", "pearson", "mae", "rmse"):
                observed = graph_eval._metric_difference(metric, metrics_a, metrics_b)
                rng = np.random.default_rng(
                    _bootstrap_seed(scheme_id, comparator, metric)
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
                    values_a = graph_eval._metric_values(
                        truth, sampled["y_pred_a"].to_numpy(dtype=float)
                    )
                    values_b = graph_eval._metric_values(
                        truth, sampled["y_pred_b"].to_numpy(dtype=float)
                    )
                    value = graph_eval._metric_difference(metric, values_a, values_b)
                    if math.isfinite(value):
                        bootstrap_values.append(value)
                if not bootstrap_values:
                    raise ResidueGeometryNeuralEvaluationError(
                        f"Bootstrap failed for {scheme_id}/{comparator}/{metric}"
                    )
                lower, upper = np.quantile(bootstrap_values, (0.025, 0.975))
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_a": MODEL_ID,
                        "model_b": comparator,
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
                progress.update(1)
                progress.set_postfix(
                    scheme=scheme_id,
                    comparator=comparator,
                    metric=metric,
                    refresh=False,
                )
    progress.close()
    return pd.DataFrame(rows)


def _random_rotation(seed: int = 33091) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return torch.tensor(q, dtype=torch.float32)


def rigid_motion_output_check() -> Mapping[str, object]:
    _configure_reproducible_execution()
    _set_seed(33090)
    device = torch.device("cpu")
    node_features = torch.randn(7, 30, device=device)
    coordinates = torch.randn(7, 3, device=device)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_features = torch.randn(edge_index.shape[1], 5, device=device)
    target_node_index = torch.tensor([1, 4], dtype=torch.long, device=device)
    substitution = torch.zeros((2, len(graph_eval.SUBSTITUTIONS)), device=device)
    substitution[0, 3] = 1.0
    substitution[1, 17] = 1.0
    condition = torch.zeros((2, 2), device=device)
    model = _model(30, 5, 2, device)
    model.eval()
    with torch.no_grad():
        reference = model(
            node_features,
            coordinates,
            edge_index,
            edge_features,
            target_node_index,
            substitution,
            condition,
        )
        rotation = _random_rotation().to(device)
        translation = torch.tensor([3.2, -7.4, 1.1], device=device)
        moved = coordinates @ rotation.T + translation
        transformed = model(
            node_features,
            moved,
            edge_index,
            edge_features,
            target_node_index,
            substitution,
            condition,
        )
    maximum_difference = float(torch.max(torch.abs(reference - transformed)))
    passed = maximum_difference <= INVARIANCE_ABSOLUTE_TOLERANCE
    return {
        "passed": passed,
        "maximum_absolute_prediction_difference": maximum_difference,
        "absolute_tolerance": INVARIANCE_ABSOLUTE_TOLERANCE,
        "transformation": "rotation_and_translation",
    }


def inspect_residue_geometry_neural_inputs(repo_root: Path) -> Mapping[str, object]:
    _verify_graph_eval_module()
    _verify_graph_eval_outputs(repo_root)
    tables = graph_eval._load_tables(repo_root)
    observations = graph_eval._build_analysis_observations(tables)
    contexts = graph_eval._make_contexts(tables, observations)

    eligibility_column = "experimental_structure_task_eligible"
    if eligibility_column not in observations.columns:
        raise ResidueGeometryNeuralEvaluationError(
            "Experimental-structure eligibility is absent from observations"
        )
    eligible = observations[observations[eligibility_column].fillna(False).astype(bool)]
    if eligible["observation_id"].nunique() != EXPECTED_STRUCTURE_OBSERVATION_COUNT:
        raise ResidueGeometryNeuralEvaluationError(
            "Unexpected structure-supported observation count"
        )
    if len(contexts) != EXPECTED_CONTEXT_COUNT:
        raise ResidueGeometryNeuralEvaluationError(
            "Unexpected evaluation-context count: "
            f"expected {EXPECTED_CONTEXT_COUNT}, observed {len(contexts)}"
        )
    if len(tables["nodes"]) != EXPECTED_GRAPH_NODE_COUNT:
        raise ResidueGeometryNeuralEvaluationError("Unexpected graph-node count")
    if len(tables["edges"]) != EXPECTED_GRAPH_EDGE_COUNT:
        raise ResidueGeometryNeuralEvaluationError("Unexpected graph-edge count")

    context_rows = []
    total_test_rows = 0
    for context in contexts:
        test = _test_rows(context)
        total_test_rows += len(test)
        record = {
            "scheme_id": str(context["scheme_id"]),
            "evaluation_iteration": str(context["evaluation_iteration"]),
            "test_row_count": len(test),
        }
        if "source_folds" in context:
            record["fitting_row_count"] = len(_fitting_rows(context))
            record["source_cv_fold_count"] = len(context["source_folds"])
        else:
            record["training_row_count"] = len(context["train"])
            record["validation_row_count"] = len(context["validation"])
        context_rows.append(record)
    if total_test_rows != EXPECTED_PREDICTIONS_PER_MODEL:
        raise ResidueGeometryNeuralEvaluationError(
            f"Unexpected evaluation-row total: {total_test_rows}"
        )

    movement_check = rigid_motion_output_check()
    if not movement_check["passed"]:
        raise ResidueGeometryNeuralEvaluationError(
            "Rigid-motion output check did not pass"
        )

    return {
        "status": "PASS",
        "analysis_scope": "residue_geometry_neural_mutation_effect_evaluation",
        "model_id": MODEL_ID,
        "evaluation_context_count": len(contexts),
        "structure_supported_observation_count": eligible["observation_id"].nunique(),
        "evaluation_prediction_count": total_test_rows,
        "graph_node_count": len(tables["nodes"]),
        "graph_edge_count": len(tables["edges"]),
        "contexts": context_rows,
        "rigid_motion_output_check": movement_check,
    }


def run_residue_geometry_neural_evaluation(
    repo_root: Path,
    device_name: str = "cuda",
    repeats: int = TRAINING_REPEATS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    postprocess_only: bool = False,
) -> Mapping[str, object]:
    _configure_reproducible_execution()
    _verify_graph_eval_module()
    _verify_graph_eval_outputs(repo_root)
    if repeats < 1 or repeats > len(FINAL_REPEAT_SEEDS):
        raise ResidueGeometryNeuralEvaluationError(
            f"repeats must be between 1 and {len(FINAL_REPEAT_SEEDS)}"
        )
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ResidueGeometryNeuralEvaluationError(
            "max_epochs, patience, and batch_size must be positive"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ResidueGeometryNeuralEvaluationError(
            "CUDA was requested but is unavailable"
        )
    device = torch.device(device_name)

    input_review = inspect_residue_geometry_neural_inputs(repo_root)
    tables = graph_eval._load_tables(repo_root)
    observations = graph_eval._build_analysis_observations(tables)
    contexts = graph_eval._make_contexts(tables, observations)

    output_dir = repo_root / "data/interim/residue_geometry_neural_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(device, repeats, max_epochs, patience, batch_size)
    checkpoint_dir = _prepare_checkpoint_directory(output_dir, contract)

    repeat_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    overall = tqdm(
        total=len(contexts),
        desc="residue geometry neural analyses",
        unit="context",
        dynamic_ncols=True,
    )
    for context in contexts:
        test_rows = _test_rows(context)
        checkpoint = _load_context_checkpoint(
            checkpoint_dir, context, test_rows, repeats
        )
        if checkpoint is not None:
            repeat_frame, history_frame, selected_epoch = checkpoint
            repeat_frames.append(repeat_frame)
            history_frames.append(history_frame)
            overall.update(1)
            overall.set_postfix(
                scheme=str(context["scheme_id"]),
                iteration=str(context["evaluation_iteration"]),
                selected_epochs=selected_epoch,
                status="checkpoint",
                refresh=False,
            )
            continue

        if postprocess_only:
            raise ResidueGeometryNeuralEvaluationError(
                "Post-processing requested but one or more context checkpoints "
                "are absent"
            )

        context_repeat_rows: list[Mapping[str, object]] = []
        context_history_rows: list[Mapping[str, object]] = []
        selected_epoch, selection_history = _selection_for_context(
            context, tables, device, max_epochs, patience, batch_size
        )
        context_history_rows.extend(selection_history)
        fitting_rows = _fitting_rows(context)
        condition_mode = graph_eval.CONDITION_MODE_BY_SCHEME[str(context["scheme_id"])]
        for repeat_index, seed in enumerate(FINAL_REPEAT_SEEDS[:repeats], start=1):
            prediction, history = _fit_fixed_epochs_and_predict(
                fitting_rows,
                test_rows,
                tables["nodes"],
                tables["edges"],
                tables["residue_features"],
                condition_mode,
                device,
                seed,
                selected_epoch,
                (
                    f"{context['scheme_id']} {context['evaluation_iteration']} "
                    f"{MODEL_ID} repeat {repeat_index}/{repeats}"
                ),
                batch_size,
            )
            context_repeat_rows.extend(
                _prediction_records(context, test_rows, prediction, repeat_index)
            )
            for row in history:
                context_history_rows.append(
                    {
                        "scheme_id": str(context["scheme_id"]),
                        "evaluation_iteration": str(context["evaluation_iteration"]),
                        "model_id": MODEL_ID,
                        "fit_role": "final_refit",
                        "source_cv_fold": "",
                        "training_repeat": repeat_index,
                        "selected_epoch_count": selected_epoch,
                        "validation_mae": np.nan,
                        **row,
                    }
                )

        _write_context_checkpoint(
            checkpoint_dir,
            context,
            context_repeat_rows,
            context_history_rows,
            selected_epoch,
            test_rows,
            repeats,
        )
        checkpoint = _load_context_checkpoint(
            checkpoint_dir, context, test_rows, repeats
        )
        if checkpoint is None:
            raise ResidueGeometryNeuralEvaluationError(
                "Context checkpoint could not be reloaded after writing"
            )
        repeat_frame, history_frame, _ = checkpoint
        repeat_frames.append(repeat_frame)
        history_frames.append(history_frame)
        overall.update(1)
        overall.set_postfix(
            scheme=str(context["scheme_id"]),
            iteration=str(context["evaluation_iteration"]),
            selected_epochs=selected_epoch,
            status="saved",
            refresh=False,
        )
    overall.close()

    repeat_predictions = pd.concat(repeat_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    predictions = _average_repeat_predictions(repeat_predictions, repeats)
    if len(predictions) != EXPECTED_PREDICTIONS_PER_MODEL:
        raise ResidueGeometryNeuralEvaluationError(
            f"Unexpected averaged prediction count: {len(predictions)}"
        )

    # Preserve the expensive fitted results before statistical comparisons.
    _write_csv(
        repeat_predictions,
        output_dir / "residue_geometry_neural_repeat_predictions.csv",
    )
    _write_csv(predictions, output_dir / "residue_geometry_neural_predictions.csv")
    _write_csv(history, output_dir / "residue_geometry_neural_training_history.csv")
    _write_json(
        {
            "status": "TRAINING_COMPLETE",
            "prediction_count": len(predictions),
            "repeat_prediction_count": len(repeat_predictions),
            "evaluation_context_count": len(contexts),
            "training_repeats": repeats,
            "repeat_predictions_sha256": _sha256(
                output_dir / "residue_geometry_neural_repeat_predictions.csv"
            ),
            "predictions_sha256": _sha256(
                output_dir / "residue_geometry_neural_predictions.csv"
            ),
            "training_history_sha256": _sha256(
                output_dir / "residue_geometry_neural_training_history.csv"
            ),
        },
        output_dir / TRAINING_COMPLETE_RECORD_NAME,
    )

    comparator_predictions = _load_comparator_predictions(repo_root)
    all_predictions = pd.concat(
        (predictions, comparator_predictions), ignore_index=True
    )
    identity = [
        "scheme_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
        "model_id",
    ]
    if all_predictions.duplicated(identity).any():
        raise ResidueGeometryNeuralEvaluationError(
            "Prediction identities are not unique after comparator merge"
        )
    if not np.isfinite(
        all_predictions[["y_true", "y_pred"]].to_numpy(dtype=float)
    ).all():
        raise ResidueGeometryNeuralEvaluationError(
            "Predictions contain non-finite values"
        )

    metrics = _metric_table(all_predictions)
    _write_csv(metrics, output_dir / "residue_geometry_neural_metrics.csv")
    paired = _paired_comparisons(all_predictions)
    _write_csv(paired, output_dir / "residue_geometry_neural_paired_comparisons.csv")

    summary = {
        "status": "PASS",
        "analysis_scope": "residue_geometry_neural_mutation_effect_evaluation",
        "model_id": MODEL_ID,
        "model_family": "E(n)-equivariant graph neural network",
        "reflection_equivariance": True,
        "stereochemistry_claimed": False,
        "primary_comparator": STAGE32_GRAPH_MODEL_ID,
        "control_model": STAGE32_CENTRE_MODEL_ID,
        "reference_models": list(REFERENCE_MODEL_IDS),
        "coordinate_source": "selected_experimental_pdb",
        "coordinate_input": (
            "C-alpha coordinates used through relative residue displacements"
        ),
        "absolute_coordinates_used_as_scalar_predictors": False,
        "protein_id_used_as_direct_model_input": False,
        "assay_position_used_as_direct_model_input": False,
        "mechanistic_reference_inputs_used": False,
        "reported_homologue_pair_inputs_used": False,
        "outer_test_labels_used_for_epoch_selection": False,
        "cross_protein_test_labels_used_for_epoch_selection": False,
        "target_outcome_used_as_model_input": False,
        "node_features_match_graph_eval": True,
        "edge_features_match_graph_eval": True,
        "condition_modes_match_graph_eval": True,
        "geometry_layer_count": GEOMETRY_LAYER_COUNT,
        "hidden_width": HIDDEN_WIDTH,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_norm_limit": GRADIENT_NORM_LIMIT,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "training_repeats": repeats,
        "selection_seed": SELECTION_SEED,
        "final_repeat_seeds": list(FINAL_REPEAT_SEEDS[:repeats]),
        "epoch_selection_measure": "validation_mae",
        "cross_protein_epoch_selection": "five_source_protein_grouped_cv_folds",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_resampling_unit": "evaluation_group_id",
        "prediction_count": len(predictions),
        "repeat_prediction_count": len(repeat_predictions),
        "evaluation_context_count": EXPECTED_CONTEXT_COUNT,
        "structure_supported_observation_count": EXPECTED_STRUCTURE_OBSERVATION_COUNT,
        "graph_node_count": EXPECTED_GRAPH_NODE_COUNT,
        "graph_edge_count": EXPECTED_GRAPH_EDGE_COUNT,
        "checkpoint_resume_supported": True,
        "postprocess_only_supported": True,
        "rigid_motion_output_check": input_review["rigid_motion_output_check"],
        "graph_eval_output_sha256": dict(STAGE32_OUTPUT_HASHES),
        "software": {
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else ""
            ),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
        },
        "outputs": [
            "residue_geometry_neural_repeat_predictions.csv",
            "residue_geometry_neural_predictions.csv",
            "residue_geometry_neural_training_history.csv",
            "residue_geometry_neural_metrics.csv",
            "residue_geometry_neural_paired_comparisons.csv",
            "residue_geometry_neural_summary.json",
        ],
    }
    _write_json(summary, output_dir / "residue_geometry_neural_summary.json")
    return summary
