from __future__ import annotations

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
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GINEConv
from tqdm.auto import tqdm

from . import residue_graph_neural_evaluation as graph_stage
from . import sequence_plm_evaluation as sequence_stage


class SequenceResidueGraphNeuralEvaluationError(RuntimeError):
    """Raised when the sequence-plus-graph analysis cannot be completed safely."""


MODEL_ID = "sequence_residue_graph_neural_regressor"
PRIMARY_COMPARATOR = graph_stage.GRAPH_MODEL
SECONDARY_REFERENCE = graph_stage.CENTRE_MODEL
ESMC_HIDDEN_STATE_INDEX = 36
ESMC_INPUT_DIMENSION = sequence_stage.ESMC_HIDDEN_SIZE
ESMC_PROJECTION_WIDTH = graph_stage.HIDDEN_WIDTH

MAX_EPOCHS = graph_stage.MAX_EPOCHS
EARLY_STOPPING_PATIENCE = graph_stage.EARLY_STOPPING_PATIENCE
BATCH_SIZE = graph_stage.BATCH_SIZE
TRAINING_REPEATS = graph_stage.TRAINING_REPEATS
SELECTION_SEED = graph_stage.SELECTION_SEED
FINAL_REPEAT_SEEDS = graph_stage.FINAL_REPEAT_SEEDS
BOOTSTRAP_REPLICATES = graph_stage.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = "beta_lactamase_sequence_residue_graph_bootstrap"
TARGET_ABSOLUTE_TOLERANCE = graph_stage.TARGET_ABSOLUTE_TOLERANCE
EXPECTED_PREDICTION_COUNT = graph_stage.EXPECTED_PREDICTIONS_PER_MODEL

OUTPUT_DIR_NAME = "sequence_residue_graph_neural_evaluation"
CHECKPOINT_DIR_NAME = "context_checkpoints"
BOOTSTRAP_DIR_NAME = "bootstrap_checkpoints"

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

SEQUENCE_SELECTION_HASHES = {
    "sequence_plm_model_selection.csv": (
        "8f843212a2b0ee7e13dfdcf439be5e582de751aa5211f527d36f521e4e9672c3"
    ),
    "sequence_plm_summary.json": (
        "982b4724500983571e403531f944779a40dd8a0e350a47fe365cc57a36978919"
    ),
}

SOURCE_GUARD_HASHES = {
    "src/betalactamase_transfer/residue_graph_neural_evaluation.py": (
        "b66d1c2873f0ae6d62aa3ddfddc24e4a6f761fc9dc9d8f8edf8d521d509e3154"
    ),
    "src/betalactamase_transfer/sequence_plm_evaluation.py": (
        "c3365bb99f7fbbc83e30807dd107f4d06c91fa0662f1bd3f60cba8b298eaf69f"
    ),
    "src/betalactamase_transfer/sequence_plm_embeddings.py": (
        "fff1bad5cc2f1f29517401f6d963c112a67274d8fe6448e5ef9198eb81a36cc9"
    ),
}


class SequenceResidueGraphRegressor(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, graph_stage.HIDDEN_WIDTH),
            nn.ReLU(),
        )
        self.convolutions = nn.ModuleList()
        self.normalisations = nn.ModuleList()
        for _ in range(graph_stage.GRAPH_LAYER_COUNT):
            message_network = nn.Sequential(
                nn.Linear(graph_stage.HIDDEN_WIDTH, graph_stage.HIDDEN_WIDTH),
                nn.ReLU(),
                nn.Linear(graph_stage.HIDDEN_WIDTH, graph_stage.HIDDEN_WIDTH),
            )
            self.convolutions.append(
                GINEConv(message_network, edge_dim=edge_input_dim, train_eps=True)
            )
            self.normalisations.append(nn.LayerNorm(graph_stage.HIDDEN_WIDTH))

        self.sequence_projection = nn.Sequential(
            nn.Linear(ESMC_INPUT_DIMENSION, ESMC_PROJECTION_WIDTH),
            nn.ReLU(),
        )
        readout_dim = (
            graph_stage.HIDDEN_WIDTH
            + ESMC_PROJECTION_WIDTH
            + len(graph_stage.SUBSTITUTIONS)
            + condition_dim
        )
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, graph_stage.HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Dropout(graph_stage.DROPOUT),
            nn.Linear(graph_stage.HIDDEN_WIDTH, graph_stage.HIDDEN_WIDTH // 2),
            nn.ReLU(),
            nn.Linear(graph_stage.HIDDEN_WIDTH // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        target_node_index: torch.Tensor,
        substitution: torch.Tensor,
        condition: torch.Tensor,
        site_delta: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.node_encoder(node_features)
        for convolution, normalisation in zip(
            self.convolutions,
            self.normalisations,
            strict=True,
        ):
            updated = convolution(hidden, edge_index, edge_features)
            hidden = normalisation(hidden + updated)
            hidden = torch.relu(hidden)
        graph_centre = hidden[target_node_index]
        sequence_centre = self.sequence_projection(site_delta)
        combined = torch.cat(
            (graph_centre, sequence_centre, substitution, condition),
            dim=1,
        )
        return self.readout(combined).squeeze(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _verify_source_guards(repo_root: Path) -> None:
    for relative, expected in SOURCE_GUARD_HASHES.items():
        path = repo_root / relative
        if not path.is_file():
            raise SequenceResidueGraphNeuralEvaluationError(
                f"Required source file is missing: {relative}"
            )
        actual = _sha256(path)
        if actual != expected:
            raise SequenceResidueGraphNeuralEvaluationError(
                "Source file differs from the reviewed sequence-plus-residue-graph "
                f"analysis handoff: {relative}"
            )


def _verify_graph_eval_outputs(repo_root: Path) -> None:
    output_dir = repo_root / "data/interim/residue_graph_neural_evaluation"
    for name, expected in STAGE32_OUTPUT_HASHES.items():
        path = output_dir / name
        if not path.is_file():
            raise SequenceResidueGraphNeuralEvaluationError(
                f"Required residue-graph neural analysis output is missing: {path}"
            )
        if _sha256(path) != expected:
            raise SequenceResidueGraphNeuralEvaluationError(
                "Residue-graph neural analysis output differs from the reviewed "
                f"result: {name}"
            )


def _verify_existing_esmc_selection(repo_root: Path) -> None:
    selection_dir = repo_root / "data/interim/sequence_plm_evaluation"
    for name, expected in SEQUENCE_SELECTION_HASHES.items():
        path = selection_dir / name
        if not path.is_file():
            raise SequenceResidueGraphNeuralEvaluationError(
                f"Required ESMC evaluation file is missing: {path}"
            )
        if _sha256(path) != expected:
            raise SequenceResidueGraphNeuralEvaluationError(
                f"ESMC evaluation file differs from the reviewed result: {name}"
            )

    summary = json.loads(
        (selection_dir / "sequence_plm_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("test_labels_used_for_model_selection") is not False:
        raise SequenceResidueGraphNeuralEvaluationError(
            "Existing ESMC selection does not establish test-label exclusion"
        )
    selection = pd.read_csv(
        selection_dir / "sequence_plm_model_selection.csv",
        dtype={"evaluation_iteration": str},
        low_memory=False,
    )
    selected = selection[
        selection["selection_scope"].astype(str).eq("selected_setting")
        & selection["selected"].astype(str).str.lower().eq("true")
    ].copy()
    if len(selected) != 17:
        raise SequenceResidueGraphNeuralEvaluationError(
            f"Expected 17 existing ESMC selected settings, observed {len(selected)}"
        )
    selected_layers = set(selected["hidden_state_index"].astype(int))
    if selected_layers != {ESMC_HIDDEN_STATE_INDEX}:
        raise SequenceResidueGraphNeuralEvaluationError(
            "Existing ESMC selected settings do not all use hidden state 36"
        )


def _load_embedding_data(
    repo_root: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    (
        observations,
        _schemes,
        _assignments,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    ) = sequence_stage._load_inputs(repo_root)
    sequence_stage._validate_inputs(
        observations,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    sequence_stage._validate_embedding_provenance(
        repo_root,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    return embedding_index, mutant_embeddings, wild_type_embeddings


def _attach_embedding_rows(
    frame: pd.DataFrame,
    embedding_index: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["protein_id", "assay_position", "wild_type", "mutant"]
    required = set(keys)
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise SequenceResidueGraphNeuralEvaluationError(
            f"Observation rows lack ESMC join fields: {missing}"
        )
    index = embedding_index[[*keys, "embedding_row"]].copy()
    if index.duplicated(keys).any():
        raise SequenceResidueGraphNeuralEvaluationError(
            "ESMC embedding index is ambiguous for protein/position/substitution"
        )
    joined = frame.merge(index, on=keys, how="left", validate="many_to_one")
    if joined["embedding_row"].isna().any():
        raise SequenceResidueGraphNeuralEvaluationError(
            "One or more sequence-plus-residue-graph analysis observations lack "
            "an ESMC embedding"
        )
    joined["embedding_row"] = joined["embedding_row"].astype(int)
    return joined


def _site_delta(
    frame: pd.DataFrame,
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> np.ndarray:
    joined = _attach_embedding_rows(frame, embedding_index)
    return sequence_stage._site_delta(
        joined,
        mutant_embeddings,
        wild_type_embeddings,
        hidden_state_index=ESMC_HIDDEN_STATE_INDEX,
    )


def _fit_embedding_transform(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != ESMC_INPUT_DIMENSION:
        raise SequenceResidueGraphNeuralEvaluationError(
            f"Unexpected ESMC site-delta shape: {values.shape}"
        )
    mean = values.mean(axis=0, dtype=np.float64)
    sd = values.std(axis=0, ddof=0, dtype=np.float64)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all():
        raise SequenceResidueGraphNeuralEvaluationError(
            "ESMC scaling values are not finite"
        )
    return mean.astype(np.float32), sd.astype(np.float32)


def _prepare_arrays(
    frame: pd.DataFrame,
    node_lookup: Mapping[tuple[str, int], int],
    condition_transform: graph_stage.ConditionTransform,
    target_mean: float,
    target_sd: float,
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    embedding_mean: np.ndarray,
    embedding_sd: np.ndarray,
) -> Mapping[str, np.ndarray]:
    base = graph_stage._prepare_observation_arrays(
        frame,
        node_lookup,
        condition_transform,
        target_mean,
        target_sd,
    )
    delta = _site_delta(
        frame,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    scaled = (delta - embedding_mean) / embedding_sd
    if not np.isfinite(scaled).all():
        raise SequenceResidueGraphNeuralEvaluationError(
            "Scaled ESMC site-delta values are not finite"
        )
    return {**base, "site_delta": scaled.astype(np.float32)}


def _loader(
    arrays: Mapping[str, np.ndarray],
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays["node_index"]).long(),
        torch.from_numpy(arrays["substitution"]).float(),
        torch.from_numpy(arrays["condition"]).float(),
        torch.from_numpy(arrays["site_delta"]).float(),
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


def _model(
    node_input_dim: int,
    edge_input_dim: int,
    condition_dim: int,
    device: torch.device,
) -> SequenceResidueGraphRegressor:
    return SequenceResidueGraphRegressor(
        node_input_dim,
        edge_input_dim,
        condition_dim,
    ).to(device)


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
    values: list[np.ndarray] = []
    with torch.no_grad():
        for node_index, substitution, condition, site_delta, _target in _loader(
            arrays,
            batch_size,
            shuffle=False,
        ):
            prediction = model(
                node_features,
                edge_index,
                edge_features,
                node_index.to(device, non_blocking=True),
                substitution.to(device, non_blocking=True),
                condition.to(device, non_blocking=True),
                site_delta.to(device, non_blocking=True),
            )
            values.append(prediction.detach().cpu().numpy())
    if not values:
        raise SequenceResidueGraphNeuralEvaluationError("Prediction batch is empty")
    output = np.concatenate(values)
    if not np.isfinite(output).all():
        raise SequenceResidueGraphNeuralEvaluationError(
            "Sequence-plus-graph predictions are not finite"
        )
    return output


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_execution() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _fit_with_validation(
    fitting_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    condition_mode: str,
    device: torch.device,
    seed: int,
    description: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[int, float, list[Mapping[str, object]]]:
    _set_seed(seed)
    condition_transform = graph_stage._fit_condition_transform(
        fitting_rows,
        condition_mode,
    )
    target_mean, target_sd = graph_stage._target_transform(fitting_rows)
    node_features, edge_index, edge_features, node_lookup = graph_stage._graph_tensors(
        tables["nodes"],
        tables["edges"],
        tables["residue_features"],
        fitting_rows,
        device,
    )
    fitting_delta = _site_delta(
        fitting_rows,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    embedding_mean, embedding_sd = _fit_embedding_transform(fitting_delta)
    fitting_arrays = _prepare_arrays(
        fitting_rows,
        node_lookup,
        condition_transform,
        target_mean,
        target_sd,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
        embedding_mean,
        embedding_sd,
    )
    validation_arrays = _prepare_arrays(
        validation_rows,
        node_lookup,
        condition_transform,
        target_mean,
        target_sd,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
        embedding_mean,
        embedding_sd,
    )
    model = _model(
        node_features.shape[1],
        edge_features.shape[1],
        condition_transform.output_dimension,
        device,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=graph_stage.LEARNING_RATE,
        weight_decay=graph_stage.WEIGHT_DECAY,
    )
    loss_function = nn.MSELoss()
    training_loader = _loader(fitting_arrays, batch_size, shuffle=True)

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
        for node_index, substitution, condition, site_delta, target in training_loader:
            optimiser.zero_grad(set_to_none=True)
            prediction = model(
                node_features,
                edge_index,
                edge_features,
                node_index.to(device, non_blocking=True),
                substitution.to(device, non_blocking=True),
                condition.to(device, non_blocking=True),
                site_delta.to(device, non_blocking=True),
            )
            target_device = target.to(device, non_blocking=True)
            loss = loss_function(prediction, target_device)
            loss.backward()
            clip_grad_norm_(model.parameters(), graph_stage.GRADIENT_NORM_LIMIT)
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
        raise SequenceResidueGraphNeuralEvaluationError(
            "Sequence-plus-graph epoch selection failed"
        )
    return best_epoch, best_validation_mae, history


def _fit_fixed_epochs_and_predict(
    fitting_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    condition_mode: str,
    device: torch.device,
    seed: int,
    epochs: int,
    description: str,
    batch_size: int,
) -> tuple[np.ndarray, list[Mapping[str, object]]]:
    _set_seed(seed)
    condition_transform = graph_stage._fit_condition_transform(
        fitting_rows,
        condition_mode,
    )
    target_mean, target_sd = graph_stage._target_transform(fitting_rows)
    node_features, edge_index, edge_features, node_lookup = graph_stage._graph_tensors(
        tables["nodes"],
        tables["edges"],
        tables["residue_features"],
        fitting_rows,
        device,
    )
    fitting_delta = _site_delta(
        fitting_rows,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    embedding_mean, embedding_sd = _fit_embedding_transform(fitting_delta)
    fitting_arrays = _prepare_arrays(
        fitting_rows,
        node_lookup,
        condition_transform,
        target_mean,
        target_sd,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
        embedding_mean,
        embedding_sd,
    )
    test_arrays = _prepare_arrays(
        test_rows,
        node_lookup,
        condition_transform,
        target_mean,
        target_sd,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
        embedding_mean,
        embedding_sd,
    )
    model = _model(
        node_features.shape[1],
        edge_features.shape[1],
        condition_transform.output_dimension,
        device,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=graph_stage.LEARNING_RATE,
        weight_decay=graph_stage.WEIGHT_DECAY,
    )
    loss_function = nn.MSELoss()
    training_loader = _loader(fitting_arrays, batch_size, shuffle=True)
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
        for node_index, substitution, condition, site_delta, target in training_loader:
            optimiser.zero_grad(set_to_none=True)
            prediction = model(
                node_features,
                edge_index,
                edge_features,
                node_index.to(device, non_blocking=True),
                substitution.to(device, non_blocking=True),
                condition.to(device, non_blocking=True),
                site_delta.to(device, non_blocking=True),
            )
            target_device = target.to(device, non_blocking=True)
            loss = loss_function(prediction, target_device)
            loss.backward()
            clip_grad_norm_(model.parameters(), graph_stage.GRADIENT_NORM_LIMIT)
            optimiser.step()
            total_squared_loss += float(loss.detach().cpu()) * len(target)
            total_rows += len(target)
        training_loss = total_squared_loss / max(1, total_rows)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_mae": np.nan,
            }
        )
        progress.set_postfix(
            train_loss=f"{training_loss:.4f}",
            refresh=False,
        )

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
        raise SequenceResidueGraphNeuralEvaluationError(
            "Final sequence-plus-graph predictions are not finite"
        )
    return prediction, history


def _selection_for_context(
    context: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[int, list[Mapping[str, object]]]:
    scheme_id = str(context["scheme_id"])
    condition_mode = graph_stage.CONDITION_MODE_BY_SCHEME[scheme_id]
    history_rows: list[Mapping[str, object]] = []
    if "source_folds" not in context:
        epoch_count, validation_mae, history = _fit_with_validation(
            context["train"],
            context["validation"],
            tables,
            embedding_index,
            mutant_embeddings,
            wild_type_embeddings,
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
                    "evaluation_iteration": context["evaluation_iteration"],
                    "model_id": MODEL_ID,
                    "fit_role": "validation_epoch_selection",
                    "source_cv_fold": "",
                    **row,
                }
            )
        history_rows.append(
            {
                "scheme_id": scheme_id,
                "evaluation_iteration": context["evaluation_iteration"],
                "model_id": MODEL_ID,
                "fit_role": "selected_epoch",
                "source_cv_fold": "",
                "epoch": epoch_count,
                "training_loss": np.nan,
                "validation_mae": validation_mae,
            }
        )
        return epoch_count, history_rows

    selected_epochs: list[int] = []
    fold_maes: list[float] = []
    for fold_record in context["source_folds"]:
        fold = int(fold_record["fold"])
        epoch_count, validation_mae, history = _fit_with_validation(
            fold_record["train"],
            fold_record["validation"],
            tables,
            embedding_index,
            mutant_embeddings,
            wild_type_embeddings,
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


def _fitting_rows_for_context(context: Mapping[str, object]) -> pd.DataFrame:
    if "source_folds" in context:
        fitting_rows = context["source"].copy()
    else:
        fitting_rows = pd.concat(
            [context["train"], context["validation"]],
            ignore_index=True,
        )

    if fitting_rows["observation_id"].astype(str).duplicated().any():
        raise SequenceResidueGraphNeuralEvaluationError(
            "Fitting observations are duplicated within an evaluation context"
        )
    return fitting_rows


def _fit_context(
    context: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    device: torch.device,
    repeats: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_epoch, selection_history = _selection_for_context(
        context,
        tables,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
        device,
        max_epochs,
        patience,
        batch_size,
    )
    scheme_id = str(context["scheme_id"])
    condition_mode = graph_stage.CONDITION_MODE_BY_SCHEME[scheme_id]
    fitting_rows = _fitting_rows_for_context(context)
    test_rows = (
        context["target"].copy()
        if "source_folds" in context
        else context["test"].copy()
    )

    reference = context["reference"].copy()
    ordered_test = reference[["observation_id"]].merge(
        test_rows,
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    if ordered_test[graph_stage.TARGET_COLUMN].isna().any():
        raise SequenceResidueGraphNeuralEvaluationError(
            "Unable to order sequence-plus-residue-graph analysis evaluation rows"
        )

    seeds = list(FINAL_REPEAT_SEEDS)
    if repeats > len(seeds):
        seeds.extend(
            FINAL_REPEAT_SEEDS[-1] + offset + 1
            for offset in range(repeats - len(seeds))
        )
    repeat_predictions: list[np.ndarray] = []
    final_history: list[Mapping[str, object]] = []
    for repeat_index in range(repeats):
        prediction, history = _fit_fixed_epochs_and_predict(
            fitting_rows,
            ordered_test,
            tables,
            embedding_index,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode,
            device,
            int(seeds[repeat_index]),
            selected_epoch,
            (
                f"{scheme_id} {context['evaluation_iteration']} {MODEL_ID} "
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
                    "model_id": MODEL_ID,
                    "fit_role": "final_refit",
                    "source_cv_fold": "",
                    "training_repeat": repeat_index + 1,
                    "selected_epoch_count": selected_epoch,
                    **row,
                }
            )

    repeat_frame, averaged = graph_stage._prediction_records(
        reference,
        MODEL_ID,
        repeat_predictions,
    )
    history_frame = pd.DataFrame([*selection_history, *final_history])
    history_frame["training_repeat"] = history_frame.get("training_repeat", "")
    history_frame["selected_epoch_count"] = history_frame.get(
        "selected_epoch_count",
        selected_epoch,
    )
    return repeat_frame, averaged, history_frame


def _context_stem(context: Mapping[str, object]) -> str:
    scheme_id = str(context["scheme_id"])
    iteration = str(context["evaluation_iteration"])
    return f"{scheme_id}__{iteration}"


def _checkpoint_paths(output_dir: Path, stem: str) -> Mapping[str, Path]:
    checkpoint_dir = output_dir / CHECKPOINT_DIR_NAME
    return {
        "complete": checkpoint_dir / f"{stem}__complete.json",
        "repeat": checkpoint_dir / f"{stem}__repeat_predictions.csv",
        "history": checkpoint_dir / f"{stem}__training_history.csv",
    }


def _load_context_checkpoint(
    output_dir: Path,
    context: Mapping[str, object],
    repeats: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    stem = _context_stem(context)
    paths = _checkpoint_paths(output_dir, stem)
    if not all(path.is_file() for path in paths.values()):
        return None
    record = json.loads(paths["complete"].read_text(encoding="utf-8"))
    expected = {
        "scheme_id": str(context["scheme_id"]),
        "evaluation_iteration": str(context["evaluation_iteration"]),
        "model_id": MODEL_ID,
        "training_repeats": repeats,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "esmc_hidden_state_index": ESMC_HIDDEN_STATE_INDEX,
        "esmc_hidden_state_selected_in_all_existing_contexts": True,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return None
    repeat_frame = pd.read_csv(
        paths["repeat"],
        dtype={"evaluation_iteration": str},
        low_memory=False,
    )
    history_frame = pd.read_csv(
        paths["history"],
        dtype={"evaluation_iteration": str},
        low_memory=False,
    )
    expected_repeat_rows = len(context["reference"]) * repeats
    if len(repeat_frame) != expected_repeat_rows:
        return None
    averaged = (
        repeat_frame.groupby(
            [*graph_stage.COMPARISON_KEY, *graph_stage.MATCHED_FIELDS, "model_id"],
            as_index=False,
            dropna=False,
            sort=False,
        )["y_pred"]
        .mean()
        .reset_index(drop=True)
    )
    if len(averaged) != len(context["reference"]):
        return None
    return repeat_frame, averaged, history_frame


def _save_context_checkpoint(
    output_dir: Path,
    context: Mapping[str, object],
    repeat_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
    repeats: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> None:
    stem = _context_stem(context)
    paths = _checkpoint_paths(output_dir, stem)
    _write_csv(repeat_frame, paths["repeat"])
    _write_csv(history_frame, paths["history"])
    record = {
        "status": "COMPLETE",
        "scheme_id": str(context["scheme_id"]),
        "evaluation_iteration": str(context["evaluation_iteration"]),
        "model_id": MODEL_ID,
        "training_repeats": repeats,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "esmc_hidden_state_index": ESMC_HIDDEN_STATE_INDEX,
        "repeat_prediction_count": len(repeat_frame),
        "repeat_predictions_sha256": _sha256(paths["repeat"]),
        "training_history_sha256": _sha256(paths["history"]),
    }
    _write_json(record, paths["complete"])


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    if len(y_true) < 2:
        return {
            "spearman": np.nan,
            "pearson": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
        }
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return graph_stage._metrics(predictions)


def _metric_difference(
    metric: str,
    a: Mapping[str, float],
    b: Mapping[str, float],
) -> float:
    if metric in {"spearman", "pearson"}:
        return float(a[metric] - b[metric])
    return float(b[metric] - a[metric])


def _bootstrap_seed(scheme_id: str) -> int:
    text = "|".join((BOOTSTRAP_SEED, scheme_id, MODEL_ID, PRIMARY_COMPARATOR))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _bootstrap_record_path(output_dir: Path, scheme_id: str) -> Path:
    return output_dir / BOOTSTRAP_DIR_NAME / f"{scheme_id}.json"


def _paired_comparisons(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []
    progress = tqdm(
        graph_stage.CORE_SCHEME_IDS,
        desc="sequence-plus-residue-graph analysis grouped bootstrap",
        unit="scheme",
        dynamic_ncols=True,
    )
    for scheme_id in progress:
        scheme = predictions[predictions["scheme_id"].astype(str).eq(scheme_id)]
        a = scheme[scheme["model_id"].eq(MODEL_ID)].copy()
        b = scheme[scheme["model_id"].eq(PRIMARY_COMPARATOR)].copy()
        paired = a[
            [*graph_stage.COMPARISON_KEY, "evaluation_group_id", "y_true", "y_pred"]
        ].merge(
            b[[*graph_stage.COMPARISON_KEY, "y_true", "y_pred"]],
            on=list(graph_stage.COMPARISON_KEY),
            how="inner",
            validate="one_to_one",
            suffixes=("_a", "_b"),
        )
        if len(paired) != len(a) or len(paired) != len(b):
            raise SequenceResidueGraphNeuralEvaluationError(
                f"Paired population mismatch for {scheme_id}"
            )
        if not np.allclose(
            paired["y_true_a"].to_numpy(dtype=float),
            paired["y_true_b"].to_numpy(dtype=float),
            rtol=0.0,
            atol=TARGET_ABSOLUTE_TOLERANCE,
        ):
            raise SequenceResidueGraphNeuralEvaluationError(
                f"Paired outcomes disagree for {scheme_id}"
            )

        y_true = paired["y_true_a"].to_numpy(dtype=float)
        pred_a = paired["y_pred_a"].to_numpy(dtype=float)
        pred_b = paired["y_pred_b"].to_numpy(dtype=float)
        metrics_a = _metric_values(y_true, pred_a)
        metrics_b = _metric_values(y_true, pred_b)
        checkpoint_path = _bootstrap_record_path(output_dir, scheme_id)
        if checkpoint_path.is_file():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            saved_rows = saved.get("rows", [])
            values_match = all(
                math.isclose(
                    float(row["model_a_value"]),
                    float(metrics_a[str(row["metric"])]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row["model_b_value"]),
                    float(metrics_b[str(row["metric"])]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for row in saved_rows
            )
            if (
                saved.get("scheme_id") == scheme_id
                and saved.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES
                and len(saved_rows) == 4
                and values_match
            ):
                rows.extend(saved_rows)
                progress.set_postfix(
                    scheme=scheme_id,
                    status="checkpoint",
                    refresh=False,
                )
                continue

        group_series = paired["evaluation_group_id"].astype(str).to_numpy()
        group_ids = np.unique(group_series)
        group_positions = {
            group_id: np.flatnonzero(group_series == group_id) for group_id in group_ids
        }
        rng = np.random.default_rng(_bootstrap_seed(scheme_id))
        bootstrap_values = {
            metric: [] for metric in ("spearman", "pearson", "mae", "rmse")
        }
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
            sampled_index = np.concatenate(
                [group_positions[group_id] for group_id in sampled_ids]
            )
            truth = y_true[sampled_index]
            values_a = _metric_values(truth, pred_a[sampled_index])
            values_b = _metric_values(truth, pred_b[sampled_index])
            for metric in bootstrap_values:
                value = _metric_difference(metric, values_a, values_b)
                if math.isfinite(value):
                    bootstrap_values[metric].append(value)

        scheme_rows = []
        for metric in ("spearman", "pearson", "mae", "rmse"):
            values = bootstrap_values[metric]
            if not values:
                raise SequenceResidueGraphNeuralEvaluationError(
                    f"No valid bootstrap replicates for {scheme_id}/{metric}"
                )
            lower, upper = np.quantile(values, [0.025, 0.975])
            scheme_rows.append(
                {
                    "scheme_id": scheme_id,
                    "model_a": MODEL_ID,
                    "model_b": PRIMARY_COMPARATOR,
                    "metric": metric,
                    "model_a_value": metrics_a[metric],
                    "model_b_value": metrics_b[metric],
                    "paired_difference": _metric_difference(
                        metric, metrics_a, metrics_b
                    ),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_valid_replicates": len(values),
                    "resampling_unit": "evaluation_group_id",
                    "difference_interpretation": "positive_favours_model_a",
                }
            )
        _write_json(
            {
                "scheme_id": scheme_id,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "rows": scheme_rows,
            },
            checkpoint_path,
        )
        rows.extend(scheme_rows)
        progress.set_postfix(
            scheme=scheme_id,
            status="complete",
            refresh=False,
        )
    return pd.DataFrame(rows)


def _graph_eval_reference_predictions(repo_root: Path) -> pd.DataFrame:
    path = (
        repo_root
        / "data/interim/residue_graph_neural_evaluation"
        / "residue_graph_neural_predictions.csv"
    )
    frame = pd.read_csv(
        path,
        dtype={"evaluation_iteration": str},
        low_memory=False,
    )
    keep = frame[
        frame["model_id"].isin((PRIMARY_COMPARATOR, SECONDARY_REFERENCE))
    ].copy()
    expected = 2 * EXPECTED_PREDICTION_COUNT
    if len(keep) != expected:
        raise SequenceResidueGraphNeuralEvaluationError(
            "Unexpected residue-graph neural analysis reference prediction count: "
            f"{len(keep)}"
        )
    return keep


def _validate_prediction_population(
    fusion: pd.DataFrame,
    graph_reference: pd.DataFrame,
) -> None:
    graph_only = graph_reference[
        graph_reference["model_id"].eq(PRIMARY_COMPARATOR)
    ].copy()
    paired = fusion[[*graph_stage.COMPARISON_KEY, "y_true"]].merge(
        graph_only[[*graph_stage.COMPARISON_KEY, "y_true"]],
        on=list(graph_stage.COMPARISON_KEY),
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_fusion", "_graph"),
    )
    if (
        not paired["_merge"].eq("both").all()
        or len(paired) != EXPECTED_PREDICTION_COUNT
    ):
        raise SequenceResidueGraphNeuralEvaluationError(
            "sequence-plus-residue-graph analysis and residue-graph neural analysis "
            "evaluation populations differ"
        )
    if not np.allclose(
        paired["y_true_fusion"].to_numpy(dtype=float),
        paired["y_true_graph"].to_numpy(dtype=float),
        rtol=0.0,
        atol=TARGET_ABSOLUTE_TOLERANCE,
    ):
        raise SequenceResidueGraphNeuralEvaluationError(
            "sequence-plus-residue-graph analysis and residue-graph neural analysis "
            "outcomes differ"
        )


def inspect_sequence_residue_graph_inputs(repo_root: Path) -> Mapping[str, object]:
    repo_root = repo_root.resolve()
    _verify_source_guards(repo_root)
    _verify_graph_eval_outputs(repo_root)
    _verify_existing_esmc_selection(repo_root)
    tables = graph_stage._load_tables(repo_root)
    graph_stage._validate_input_contract(tables)
    observations = graph_stage._build_analysis_observations(tables)
    contexts = graph_stage._make_contexts(tables, observations)
    embedding_index, mutant_embeddings, wild_type_embeddings = _load_embedding_data(
        repo_root
    )

    context_records = []
    evaluation_records = []
    all_evaluation_ids: set[str] = set()
    all_fitting_ids: set[str] = set()
    total_missing_esmc_evaluation_rows = 0
    for context in contexts:
        fitting_rows = _fitting_rows_for_context(context)
        test_rows = context["target"] if "source_folds" in context else context["test"]

        if test_rows["observation_id"].astype(str).duplicated().any():
            raise SequenceResidueGraphNeuralEvaluationError(
                "Evaluation observations are duplicated within an evaluation context"
            )

        fitting = _attach_embedding_rows(fitting_rows, embedding_index)
        testing = _attach_embedding_rows(test_rows, embedding_index)
        fitting_delta = _site_delta(
            fitting_rows,
            embedding_index,
            mutant_embeddings,
            wild_type_embeddings,
        )
        testing_delta = _site_delta(
            test_rows,
            embedding_index,
            mutant_embeddings,
            wild_type_embeddings,
        )
        if not np.isfinite(fitting_delta).all() or not np.isfinite(testing_delta).all():
            raise SequenceResidueGraphNeuralEvaluationError(
                "One or more ESMC values are not finite"
            )
        fit_ids = set(fitting["observation_id"].astype(str))
        test_ids = set(testing["observation_id"].astype(str))
        if fit_ids.intersection(test_ids):
            raise SequenceResidueGraphNeuralEvaluationError(
                "Fitting and evaluation observations overlap"
            )
        all_fitting_ids.update(fit_ids)
        all_evaluation_ids.update(test_ids)
        reference = context["reference"]
        reference_ids = set(reference["observation_id"].astype(str))
        if test_ids != reference_ids:
            raise SequenceResidueGraphNeuralEvaluationError(
                "Evaluation observations differ from the recorded reference population"
            )
        evaluation_records.append(
            reference[[*graph_stage.COMPARISON_KEY, "y_true"]].copy()
        )
        missing_esmc_rows = int(testing["embedding_row"].isna().sum())
        total_missing_esmc_evaluation_rows += missing_esmc_rows
        record = {
            "scheme_id": str(context["scheme_id"]),
            "evaluation_iteration": str(context["evaluation_iteration"]),
            "fitting_row_count": len(fitting),
            "test_row_count": len(testing),
            "missing_esmc_rows": missing_esmc_rows,
        }
        if "source_folds" in context:
            record["source_cv_fold_count"] = len(context["source_folds"])
        context_records.append(record)

    evaluation_frame = pd.concat(evaluation_records, ignore_index=True)
    if len(evaluation_frame) != EXPECTED_PREDICTION_COUNT:
        raise SequenceResidueGraphNeuralEvaluationError(
            "sequence-plus-residue-graph analysis evaluation context-row count "
            "differs from residue-graph neural analysis"
        )
    if evaluation_frame.duplicated(list(graph_stage.COMPARISON_KEY)).any():
        raise SequenceResidueGraphNeuralEvaluationError(
            "sequence-plus-residue-graph analysis evaluation context identities "
            "are duplicated"
        )

    graph_eval = _graph_eval_reference_predictions(repo_root)
    _validate_prediction_population(evaluation_frame, graph_eval)
    graph_only = graph_eval[graph_eval["model_id"].eq(PRIMARY_COMPARATOR)].copy()
    graph_eval_ids = set(graph_only["observation_id"].astype(str))
    if all_evaluation_ids != graph_eval_ids:
        raise SequenceResidueGraphNeuralEvaluationError(
            "sequence-plus-residue-graph analysis unique evaluation observations "
            "do not match residue-graph neural analysis"
        )

    condition_dim = max(
        graph_stage._fit_condition_transform(
            (context["source"] if "source_folds" in context else context["train"]),
            graph_stage.CONDITION_MODE_BY_SCHEME[str(context["scheme_id"])],
        ).output_dimension
        for context in contexts
    )
    node_dim = (
        len(graph_stage.AMINO_ACIDS)
        + len(graph_stage.SECONDARY_STRUCTURE_STATES)
        + len(graph_stage.CENTRE_NUMERIC_COLUMNS)
    )
    edge_dim = 5
    parameter_count = sum(
        parameter.numel()
        for parameter in _model(
            node_dim,
            edge_dim,
            condition_dim,
            torch.device("cpu"),
        ).parameters()
    )
    return {
        "status": "PASS",
        "analysis_scope": "sequence_residue_graph_neural_mutation_effect_evaluation",
        "model_id": MODEL_ID,
        "primary_comparator": PRIMARY_COMPARATOR,
        "esmc_model_id": sequence_stage.ESMC_SOURCE_MODEL_ID,
        "esmc_model_revision": sequence_stage.ESMC_SOURCE_MODEL_REVISION,
        "esmc_hidden_state_index": ESMC_HIDDEN_STATE_INDEX,
        "esmc_site_delta_dimension": ESMC_INPUT_DIMENSION,
        "esmc_projection_width": ESMC_PROJECTION_WIDTH,
        "evaluation_context_count": len(contexts),
        "evaluation_prediction_count": len(evaluation_frame),
        "unique_evaluation_observation_count": len(all_evaluation_ids),
        "unique_fitting_observations_across_contexts": len(all_fitting_ids),
        "missing_esmc_evaluation_rows": total_missing_esmc_evaluation_rows,
        "graph_eval_population_match": True,
        "test_labels_used_for_model_selection": False,
        "cross_protein_target_labels_used_for_model_selection": False,
        "embedding_scaling_population": "fitting rows only",
        "graph_inputs_match_graph_eval": True,
        "training_settings_match_graph_eval": True,
        "selection_seed_matches_graph_eval": True,
        "final_repeat_seeds_match_graph_eval": True,
        "parameter_count_at_max_condition_dimension": int(parameter_count),
        "contexts": context_records,
    }


def prepare_sequence_residue_graph_neural_evaluation(
    repo_root: Path,
    *,
    device_name: str = "cuda",
    repeats: int = TRAINING_REPEATS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    postprocess_only: bool = False,
) -> Mapping[str, object]:
    repo_root = repo_root.resolve()
    if repeats < 1 or max_epochs < 1 or patience < 1 or batch_size < 1:
        raise SequenceResidueGraphNeuralEvaluationError(
            "Training settings must be positive"
        )
    _configure_torch_execution()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SequenceResidueGraphNeuralEvaluationError("CUDA is not available")
    device = torch.device(device_name)
    audit = inspect_sequence_residue_graph_inputs(repo_root)
    tables = graph_stage._load_tables(repo_root)
    observations = graph_stage._build_analysis_observations(tables)
    contexts = graph_stage._make_contexts(tables, observations)
    embedding_index, mutant_embeddings, wild_type_embeddings = _load_embedding_data(
        repo_root
    )
    output_dir = repo_root / "data/interim" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / "sequence_residue_graph_predictions.csv"
    repeat_path = output_dir / "sequence_residue_graph_repeat_predictions.csv"
    history_path = output_dir / "sequence_residue_graph_training_history.csv"
    training_complete_path = output_dir / "training_complete.json"

    if postprocess_only:
        required = (prediction_path, repeat_path, history_path, training_complete_path)
        if not all(path.is_file() for path in required):
            raise SequenceResidueGraphNeuralEvaluationError(
                "Postprocessing requires completed sequence-plus-residue-graph "
                "analysis training files"
            )
        predictions = pd.read_csv(
            prediction_path,
            dtype={"evaluation_iteration": str},
            low_memory=False,
        )
        repeat_predictions = pd.read_csv(
            repeat_path,
            dtype={"evaluation_iteration": str},
            low_memory=False,
        )
        training_history = pd.read_csv(
            history_path,
            dtype={"evaluation_iteration": str},
            low_memory=False,
        )
    else:
        repeat_frames = []
        prediction_frames = []
        history_frames = []
        progress = tqdm(
            contexts,
            desc="sequence plus residue graph analyses",
            unit="context",
            dynamic_ncols=True,
        )
        for context in progress:
            progress.set_postfix(
                scheme=str(context["scheme_id"]),
                iteration=str(context["evaluation_iteration"]),
                refresh=False,
            )
            saved = _load_context_checkpoint(
                output_dir,
                context,
                repeats,
                max_epochs,
                patience,
                batch_size,
            )
            if saved is None:
                repeat_frame, averaged, history_frame = _fit_context(
                    context,
                    tables,
                    embedding_index,
                    mutant_embeddings,
                    wild_type_embeddings,
                    device,
                    repeats,
                    max_epochs,
                    patience,
                    batch_size,
                )
                _save_context_checkpoint(
                    output_dir,
                    context,
                    repeat_frame,
                    history_frame,
                    repeats,
                    max_epochs,
                    patience,
                    batch_size,
                )
            else:
                repeat_frame, averaged, history_frame = saved
                progress.set_postfix(
                    scheme=str(context["scheme_id"]),
                    iteration=str(context["evaluation_iteration"]),
                    status="checkpoint",
                    refresh=False,
                )
            repeat_frames.append(repeat_frame)
            prediction_frames.append(averaged)
            history_frames.append(history_frame)
        predictions = pd.concat(prediction_frames, ignore_index=True)
        repeat_predictions = pd.concat(repeat_frames, ignore_index=True)
        training_history = pd.concat(history_frames, ignore_index=True)
        _write_csv(predictions, prediction_path)
        _write_csv(repeat_predictions, repeat_path)
        _write_csv(training_history, history_path)
        training_complete = {
            "status": "TRAINING_COMPLETE",
            "evaluation_context_count": len(contexts),
            "prediction_count": len(predictions),
            "repeat_prediction_count": len(repeat_predictions),
            "training_repeats": repeats,
            "predictions_sha256": _sha256(prediction_path),
            "repeat_predictions_sha256": _sha256(repeat_path),
            "training_history_sha256": _sha256(history_path),
        }
        _write_json(training_complete, training_complete_path)

    if len(predictions) != EXPECTED_PREDICTION_COUNT:
        raise SequenceResidueGraphNeuralEvaluationError(
            "Unexpected sequence-plus-residue-graph analysis prediction count: "
            f"{len(predictions)}"
        )
    if len(repeat_predictions) != EXPECTED_PREDICTION_COUNT * repeats:
        raise SequenceResidueGraphNeuralEvaluationError(
            "Unexpected sequence-plus-residue-graph analysis repeat-prediction "
            f"count: {len(repeat_predictions)}"
        )
    if predictions.duplicated([*graph_stage.COMPARISON_KEY, "model_id"]).any():
        raise SequenceResidueGraphNeuralEvaluationError(
            "sequence-plus-residue-graph analysis prediction identities are duplicated"
        )

    graph_eval_reference = _graph_eval_reference_predictions(repo_root)
    _validate_prediction_population(predictions, graph_eval_reference)
    combined = pd.concat([predictions, graph_eval_reference], ignore_index=True)
    metrics = _metrics(combined)
    paired = _paired_comparisons(combined, output_dir)

    metrics_path = output_dir / "sequence_residue_graph_metrics.csv"
    paired_path = output_dir / "sequence_residue_graph_paired_comparisons.csv"
    summary_path = output_dir / "sequence_residue_graph_summary.json"
    _write_csv(metrics, metrics_path)
    _write_csv(paired, paired_path)

    summary = {
        "status": "PASS",
        "analysis_scope": audit["analysis_scope"],
        "model_id": MODEL_ID,
        "primary_comparator": PRIMARY_COMPARATOR,
        "secondary_reference": SECONDARY_REFERENCE,
        "prediction_count": len(predictions),
        "repeat_prediction_count": len(repeat_predictions),
        "training_repeats": repeats,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_resampling_unit": "evaluation_group_id",
        "esmc_model_id": sequence_stage.ESMC_SOURCE_MODEL_ID,
        "esmc_model_revision": sequence_stage.ESMC_SOURCE_MODEL_REVISION,
        "esmc_hidden_state_index": ESMC_HIDDEN_STATE_INDEX,
        "esmc_site_delta_definition": (
            "mutant-site ESMC hidden state minus the corresponding wild-type "
            "residue hidden state"
        ),
        "esmc_site_delta_dimension": ESMC_INPUT_DIMENSION,
        "esmc_projection_width": ESMC_PROJECTION_WIDTH,
        "embedding_scaling_population": "fitting rows only",
        "graph_operator": "GINEConv",
        "graph_layer_count": graph_stage.GRAPH_LAYER_COUNT,
        "hidden_width": graph_stage.HIDDEN_WIDTH,
        "dropout": graph_stage.DROPOUT,
        "learning_rate": graph_stage.LEARNING_RATE,
        "weight_decay": graph_stage.WEIGHT_DECAY,
        "gradient_norm_limit": graph_stage.GRADIENT_NORM_LIMIT,
        "selection_seed": SELECTION_SEED,
        "final_repeat_seeds": list(FINAL_REPEAT_SEEDS[:repeats]),
        "test_labels_used_for_model_selection": False,
        "cross_protein_target_labels_used_for_model_selection": False,
        "protein_id_used_as_direct_model_input": False,
        "assay_position_used_as_direct_model_input": False,
        "mechanistic_reference_inputs_used": False,
        "reported_homologue_pair_inputs_used": False,
        "graph_eval_population_match": True,
        "checkpoint_resume_supported": True,
        "postprocess_only_supported": True,
        "outputs": [
            prediction_path.name,
            repeat_path.name,
            history_path.name,
            metrics_path.name,
            paired_path.name,
            summary_path.name,
        ],
        "graph_eval_output_sha256": STAGE32_OUTPUT_HASHES,
        "sequence_selection_sha256": SEQUENCE_SELECTION_HASHES,
        "source_guard_sha256": SOURCE_GUARD_HASHES,
        "software": {
            "device": str(device),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
        },
    }
    _write_json(summary, summary_path)
    return summary


def run_sequence_residue_graph_neural_evaluation(
    repo_root: Path,
    *,
    device_name: str = "cuda",
    repeats: int = TRAINING_REPEATS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    postprocess_only: bool = False,
) -> Mapping[str, object]:
    return prepare_sequence_residue_graph_neural_evaluation(
        repo_root,
        device_name=device_name,
        repeats=repeats,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        postprocess_only=postprocess_only,
    )
