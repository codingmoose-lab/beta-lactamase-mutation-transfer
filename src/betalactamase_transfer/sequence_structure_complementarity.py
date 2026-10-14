from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from . import residue_graph_neural_evaluation as graph_stage
from . import sequence_plm_evaluation as sequence_stage
from . import sequence_residue_graph_neural_evaluation as combined_stage


class SequenceStructureComplementarityError(RuntimeError):
    pass


MODEL_ID = "validation_selected_sequence_graph_combination"
SEQUENCE_MODEL = sequence_stage.MODEL_ID
GRAPH_MODEL = graph_stage.GRAPH_MODEL
COMBINED_MODEL = combined_stage.MODEL_ID
DIAGNOSTIC_ORACLE_MODEL = "diagnostic_oracle_sequence_or_graph"
OUTPUT_DIR_NAME = "sequence_structure_complementarity"
VALIDATION_CHECKPOINT_DIR = "validation_checkpoints"
GRAPH_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_sequence_structure_complementarity_bootstrap"
EXPECTED_FINAL_ROWS = graph_stage.EXPECTED_PREDICTIONS_PER_MODEL
TARGET_ABSOLUTE_TOLERANCE = graph_stage.TARGET_ABSOLUTE_TOLERANCE

SOURCE_GUARDS = {
    "src/betalactamase_transfer/sequence_plm_evaluation.py": (
        "c3365bb99f7fbbc83e30807dd107f4d06c91fa0662f1bd3f60cba8b298eaf69f"
    ),
    "src/betalactamase_transfer/residue_graph_neural_evaluation.py": (
        "b66d1c2873f0ae6d62aa3ddfddc24e4a6f761fc9dc9d8f8edf8d521d509e3154"
    ),
    "src/betalactamase_transfer/sequence_residue_graph_neural_evaluation.py": (
        "d8ecdb1889430439aa4ab2234e57055900632d31f2b6cf982f893894d0aedb42"
    ),
}
RESULT_GUARDS = {
    "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv": (
        "a811b884f18db4cb16dae1599e7657051e064db8b3d270d6e8ed2f8b0eef3660"
    ),
    "data/interim/sequence_plm_evaluation/sequence_plm_model_selection.csv": (
        "8f843212a2b0ee7e13dfdcf439be5e582de751aa5211f527d36f521e4e9672c3"
    ),
    (
        "data/interim/residue_graph_neural_evaluation/"
        "residue_graph_neural_predictions.csv"
    ): "98aa268232824720aecd2a8510d4169e4faba4f4dfd71c31659b92088af0203a",
    (
        "data/interim/residue_graph_neural_evaluation/"
        "residue_graph_neural_training_history.csv"
    ): "7b5ca9d0bb055636c21a44dc6576e94b951ab9876ab54d733d81b34025ed60e4",
    (
        "data/interim/sequence_residue_graph_neural_evaluation/"
        "sequence_residue_graph_predictions.csv"
    ): "6d6ddae3f9654b8dd0a79dd750ede8755b2ac35bb3f8454d0e4d3457796fed80",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_guards(repo_root: Path) -> None:
    for relative, expected in {**SOURCE_GUARDS, **RESULT_GUARDS}.items():
        path = repo_root / relative
        if not path.is_file():
            raise SequenceStructureComplementarityError(
                f"Required file is absent: {relative}"
            )
        observed = _sha256(path)
        if observed != expected:
            raise SequenceStructureComplementarityError(
                f"Source/result identity changed for {relative}: {observed}"
            )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"evaluation_iteration": str}, low_memory=False)


def _selected_sequence_settings(repo_root: Path) -> pd.DataFrame:
    table = _read_csv(
        repo_root
        / "data/interim/sequence_plm_evaluation/sequence_plm_model_selection.csv"
    )
    selected = table[
        table["selection_scope"].astype(str).eq("selected_setting")
        & table["selected"].astype(str).str.lower().eq("true")
    ].copy()
    if len(selected) != 17:
        raise SequenceStructureComplementarityError(
            f"Expected 17 selected ESMC settings, observed {len(selected)}"
        )
    key = ["scheme_id", "evaluation_iteration"]
    if selected.duplicated(key).any():
        raise SequenceStructureComplementarityError(
            "Selected ESMC settings are duplicated"
        )
    return selected


def _selected_graph_epochs(repo_root: Path) -> pd.DataFrame:
    table = _read_csv(
        repo_root
        / (
            "data/interim/residue_graph_neural_evaluation/"
            "residue_graph_neural_training_history.csv"
        )
    )
    selected = table[
        table["model_id"].astype(str).eq(GRAPH_MODEL)
        & table["fit_role"].astype(str).eq("selected_epoch")
    ].copy()
    if len(selected) != 17:
        raise SequenceStructureComplementarityError(
            f"Expected 17 selected graph epochs, observed {len(selected)}"
        )
    selected["selected_epoch"] = pd.to_numeric(
        selected["epoch"], errors="raise"
    ).astype(int)
    if (selected["selected_epoch"] < 1).any():
        raise SequenceStructureComplementarityError(
            "Selected graph epochs must be positive"
        )
    key = ["scheme_id", "evaluation_iteration"]
    if selected.duplicated(key).any():
        raise SequenceStructureComplementarityError(
            "Selected graph epochs are duplicated"
        )
    return selected


def _lookup_setting(table: pd.DataFrame, scheme_id: str, iteration: str) -> pd.Series:
    rows = table[
        table["scheme_id"].astype(str).eq(scheme_id)
        & table["evaluation_iteration"].astype(str).eq(str(iteration))
    ]
    if len(rows) != 1:
        raise SequenceStructureComplementarityError(
            "Expected one selected setting for "
            f"{scheme_id}/{iteration}, observed {len(rows)}"
        )
    return rows.iloc[0]


def _sequence_prediction(
    fitting_rows: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    setting: pd.Series,
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    condition_mode: str,
) -> np.ndarray:
    fitting = combined_stage._attach_embedding_rows(fitting_rows, embedding_index)
    prediction = combined_stage._attach_embedding_rows(prediction_rows, embedding_index)
    return sequence_stage._fit_predict(
        fitting,
        prediction,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
        hidden_state_index=int(setting["hidden_state_index"]),
        alpha=float(setting["alpha"]),
    )


def _graph_prediction(
    fitting_rows: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    selected_epoch: int,
    tables: Mapping[str, pd.DataFrame],
    condition_mode: str,
    device: torch.device,
    description: str,
    batch_size: int,
) -> np.ndarray:
    seed_predictions = []
    for seed_index, seed in enumerate(graph_stage.FINAL_REPEAT_SEEDS, start=1):
        prediction, _history = graph_stage._fit_fixed_epochs_and_predict(
            GRAPH_MODEL,
            fitting_rows,
            prediction_rows,
            tables["nodes"],
            tables["edges"],
            tables["residue_features"],
            condition_mode,
            device,
            int(seed),
            int(selected_epoch),
            (
                f"{description} training seed "
                f"{seed_index}/{len(graph_stage.FINAL_REPEAT_SEEDS)}"
            ),
            batch_size,
        )
        seed_predictions.append(prediction)
    return np.mean(np.stack(seed_predictions, axis=0), axis=0)


def _validation_frame(
    scheme_id: str,
    iteration: str,
    role: str,
    rows: pd.DataFrame,
    group_column: str,
    sequence_prediction: np.ndarray,
    graph_prediction: np.ndarray,
    sequence_setting: pd.Series,
    graph_epoch: int,
) -> pd.DataFrame:
    if len(rows) != len(sequence_prediction) or len(rows) != len(graph_prediction):
        raise SequenceStructureComplementarityError(
            "Validation prediction lengths disagree"
        )
    if group_column not in rows.columns:
        raise SequenceStructureComplementarityError(
            f"Validation rows lack scheme grouping column: {group_column}"
        )
    if rows[group_column].isna().any():
        raise SequenceStructureComplementarityError(
            f"Validation rows contain missing values in {group_column}"
        )
    return pd.DataFrame(
        {
            "scheme_id": scheme_id,
            "evaluation_iteration": str(iteration),
            "validation_role": role,
            "observation_id": rows["observation_id"].astype(str).to_numpy(),
            "evaluation_group_id": rows[group_column].astype(str).to_numpy(),
            "y_true": rows[graph_stage.TARGET_COLUMN].to_numpy(dtype=float),
            "sequence_prediction": sequence_prediction,
            "graph_prediction": graph_prediction,
            "sequence_hidden_state_index": int(sequence_setting["hidden_state_index"]),
            "sequence_alpha": float(sequence_setting["alpha"]),
            "graph_epoch": int(graph_epoch),
            "graph_training_seed_count": len(graph_stage.FINAL_REPEAT_SEEDS),
        }
    )


def _checkpoint_path(output_dir: Path, scheme_id: str, token: str) -> Path:
    safe = token.replace("/", "_")
    return output_dir / VALIDATION_CHECKPOINT_DIR / f"{scheme_id}__{safe}.csv"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _validate_validation_checkpoint(
    frame: pd.DataFrame,
    expected_rows: pd.DataFrame,
    scheme_id: str,
    iteration: str,
    role: str,
    group_column: str,
    sequence_setting: pd.Series,
    graph_epoch: int,
) -> None:
    required = {
        "scheme_id",
        "evaluation_iteration",
        "validation_role",
        "observation_id",
        "evaluation_group_id",
        "y_true",
        "sequence_prediction",
        "graph_prediction",
        "sequence_hidden_state_index",
        "sequence_alpha",
        "graph_epoch",
        "graph_training_seed_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SequenceStructureComplementarityError(
            f"Validation checkpoint lacks columns: {missing}"
        )
    if not frame["scheme_id"].astype(str).eq(scheme_id).all():
        raise SequenceStructureComplementarityError(
            "Validation checkpoint scheme differs"
        )
    if not frame["evaluation_iteration"].astype(str).eq(str(iteration)).all():
        raise SequenceStructureComplementarityError(
            "Validation checkpoint iteration differs"
        )
    if not frame["validation_role"].astype(str).eq(role).all():
        raise SequenceStructureComplementarityError(
            "Validation checkpoint role differs"
        )
    expected_ids = expected_rows["observation_id"].astype(str).tolist()
    observed_ids = frame["observation_id"].astype(str).tolist()
    if observed_ids != expected_ids:
        raise SequenceStructureComplementarityError(
            "Validation checkpoint observation order or population differs"
        )
    if group_column not in expected_rows.columns:
        raise SequenceStructureComplementarityError(
            f"Current validation rows lack scheme grouping column: {group_column}"
        )
    if expected_rows[group_column].isna().any():
        raise SequenceStructureComplementarityError(
            f"Current validation rows contain missing values in {group_column}"
        )
    expected_groups = expected_rows[group_column].astype(str).tolist()
    observed_groups = frame["evaluation_group_id"].astype(str).tolist()
    if observed_groups != expected_groups:
        raise SequenceStructureComplementarityError(
            "Validation checkpoint grouping identities differ from "
            "the current partition"
        )
    expected_y = expected_rows[graph_stage.TARGET_COLUMN].to_numpy(dtype=float)
    observed_y = frame["y_true"].to_numpy(dtype=float)
    if not np.allclose(
        expected_y, observed_y, rtol=0.0, atol=TARGET_ABSOLUTE_TOLERANCE
    ):
        raise SequenceStructureComplementarityError(
            "Validation checkpoint outcomes differ from the current partition"
        )
    if not np.isfinite(
        frame[["sequence_prediction", "graph_prediction"]].to_numpy(dtype=float)
    ).all():
        raise SequenceStructureComplementarityError(
            "Validation checkpoint contains non-finite predictions"
        )
    if set(frame["sequence_hidden_state_index"].astype(int)) != {
        int(sequence_setting["hidden_state_index"])
    }:
        raise SequenceStructureComplementarityError(
            "Validation checkpoint ESMC hidden-state setting differs"
        )
    if not np.allclose(
        frame["sequence_alpha"].to_numpy(dtype=float),
        float(sequence_setting["alpha"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise SequenceStructureComplementarityError(
            "Validation checkpoint ESMC alpha differs"
        )
    if set(frame["graph_epoch"].astype(int)) != {int(graph_epoch)}:
        raise SequenceStructureComplementarityError(
            "Validation checkpoint graph epoch differs"
        )
    if set(frame["graph_training_seed_count"].astype(int)) != {
        len(graph_stage.FINAL_REPEAT_SEEDS)
    }:
        raise SequenceStructureComplementarityError(
            "Validation checkpoint graph training-seed count differs"
        )


def _select_weight(rows: pd.DataFrame, selection_id: str) -> tuple[pd.DataFrame, float]:
    y_true = rows["y_true"].to_numpy(dtype=float)
    seq = rows["sequence_prediction"].to_numpy(dtype=float)
    graph = rows["graph_prediction"].to_numpy(dtype=float)
    records = []
    for graph_weight in GRAPH_WEIGHTS:
        pred = graph_weight * graph + (1.0 - graph_weight) * seq
        mae = float(np.mean(np.abs(y_true - pred)))
        records.append(
            {
                "selection_id": selection_id,
                "graph_weight": graph_weight,
                "sequence_weight": 1.0 - graph_weight,
                "validation_mae": mae,
            }
        )
    table = pd.DataFrame(records)
    chosen = min(
        records,
        key=lambda record: (record["validation_mae"], -record["graph_weight"]),
    )
    table["selected"] = table["graph_weight"].eq(chosen["graph_weight"])
    return table, float(chosen["graph_weight"])


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    return graph_stage._metric_values(y_true, y_pred)


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return graph_stage._metrics(predictions)


def _bootstrap_seed(scheme_id: str, model_b: str, metric: str) -> int:
    text = "|".join((BOOTSTRAP_SEED, scheme_id, MODEL_ID, model_b, metric))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _paired_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme_id in graph_stage.CORE_SCHEME_IDS:
        scheme = predictions[predictions["scheme_id"].astype(str).eq(scheme_id)]
        a = scheme[scheme["model_id"].eq(MODEL_ID)].copy()
        for model_b in (GRAPH_MODEL, SEQUENCE_MODEL, COMBINED_MODEL):
            b = scheme[scheme["model_id"].eq(model_b)].copy()
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
                raise SequenceStructureComplementarityError(
                    f"Final comparison population mismatch for {scheme_id}/{model_b}"
                )
            if not np.allclose(
                paired["y_true_a"],
                paired["y_true_b"],
                rtol=0.0,
                atol=TARGET_ABSOLUTE_TOLERANCE,
            ):
                raise SequenceStructureComplementarityError(
                    f"Final comparison outcomes disagree for {scheme_id}/{model_b}"
                )
            truth = paired["y_true_a"].to_numpy(dtype=float)
            pred_a = paired["y_pred_a"].to_numpy(dtype=float)
            pred_b = paired["y_pred_b"].to_numpy(dtype=float)
            metrics_a = _metric_values(truth, pred_a)
            metrics_b = _metric_values(truth, pred_b)
            group_ids = paired["evaluation_group_id"].astype(str).to_numpy()
            unique_groups = np.unique(group_ids)
            group_indices = {g: np.flatnonzero(group_ids == g) for g in unique_groups}
            for metric in ("spearman", "pearson", "mae", "rmse"):
                observed = graph_stage._metric_difference(metric, metrics_a, metrics_b)
                rng = np.random.default_rng(_bootstrap_seed(scheme_id, model_b, metric))
                values = []
                for _ in range(BOOTSTRAP_REPLICATES):
                    sampled_groups = rng.choice(
                        unique_groups, size=len(unique_groups), replace=True
                    )
                    indices = np.concatenate([group_indices[g] for g in sampled_groups])
                    m_a = _metric_values(truth[indices], pred_a[indices])
                    m_b = _metric_values(truth[indices], pred_b[indices])
                    value = graph_stage._metric_difference(metric, m_a, m_b)
                    if math.isfinite(value):
                        values.append(value)
                if not values:
                    raise SequenceStructureComplementarityError(
                        f"No valid bootstrap values for {scheme_id}/{model_b}/{metric}"
                    )
                lower, upper = np.quantile(values, [0.025, 0.975])
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_a": MODEL_ID,
                        "model_b": model_b,
                        "metric": metric,
                        "model_a_value": metrics_a[metric],
                        "model_b_value": metrics_b[metric],
                        "paired_difference": observed,
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_valid_replicates": len(values),
                        "resampling_unit": "evaluation_group_id",
                        "difference_interpretation": "positive_favours_model_a",
                    }
                )
    return pd.DataFrame(rows)


def _final_reference_predictions(repo_root: Path) -> pd.DataFrame:
    graph = _read_csv(
        repo_root
        / (
            "data/interim/residue_graph_neural_evaluation/"
            "residue_graph_neural_predictions.csv"
        )
    )
    graph = graph[graph["model_id"].astype(str).eq(GRAPH_MODEL)].copy()
    sequence = _read_csv(
        repo_root / "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv"
    )
    sequence = sequence[sequence["model_id"].astype(str).eq(SEQUENCE_MODEL)].copy()
    combined = _read_csv(
        repo_root
        / (
            "data/interim/sequence_residue_graph_neural_evaluation/"
            "sequence_residue_graph_predictions.csv"
        )
    )
    combined = combined[combined["model_id"].astype(str).eq(COMBINED_MODEL)].copy()
    if len(graph) != EXPECTED_FINAL_ROWS or len(combined) != EXPECTED_FINAL_ROWS:
        raise SequenceStructureComplementarityError(
            "Graph or combined final population is incomplete"
        )
    base = graph.copy()
    key = list(graph_stage.COMPARISON_KEY)
    for label, frame in (("sequence", sequence), ("combined", combined)):
        subset = frame[[*key, "y_true", "y_pred"]].rename(
            columns={"y_true": f"y_true_{label}", "y_pred": f"y_pred_{label}"}
        )
        base = base.merge(subset, on=key, how="left", validate="one_to_one")
        if base[f"y_pred_{label}"].isna().any():
            raise SequenceStructureComplementarityError(
                f"Missing {label} final predictions"
            )
        if not np.allclose(
            base["y_true"],
            base[f"y_true_{label}"],
            rtol=0.0,
            atol=TARGET_ABSOLUTE_TOLERANCE,
        ):
            raise SequenceStructureComplementarityError(
                f"{label} final outcomes disagree"
            )
    return base


def _diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme_id in graph_stage.CORE_SCHEME_IDS:
        scheme = predictions[predictions["scheme_id"].astype(str).eq(scheme_id)]
        wide = None
        for model in (SEQUENCE_MODEL, GRAPH_MODEL, COMBINED_MODEL, MODEL_ID):
            current = scheme[scheme["model_id"].eq(model)][
                [*graph_stage.COMPARISON_KEY, "y_true", "y_pred"]
            ].rename(columns={"y_true": f"y_true_{model}", "y_pred": model})
            wide = (
                current
                if wide is None
                else wide.merge(
                    current,
                    on=list(graph_stage.COMPARISON_KEY),
                    how="inner",
                    validate="one_to_one",
                )
            )
        truth = wide[f"y_true_{MODEL_ID}"].to_numpy(dtype=float)
        seq = wide[SEQUENCE_MODEL].to_numpy(dtype=float)
        graph = wide[GRAPH_MODEL].to_numpy(dtype=float)
        oracle = np.where(np.abs(truth - seq) < np.abs(truth - graph), seq, graph)
        residual_seq = truth - seq
        residual_graph = truth - graph
        residual_pearson = float(np.corrcoef(residual_seq, residual_graph)[0, 1])
        residual_spearman = float(
            pd.Series(residual_seq).corr(pd.Series(residual_graph), method="spearman")
        )
        absolute_errors = {
            model: np.abs(truth - wide[model].to_numpy(dtype=float))
            for model in (SEQUENCE_MODEL, GRAPH_MODEL, COMBINED_MODEL, MODEL_ID)
        }
        stacked = np.column_stack(list(absolute_errors.values()))
        minimum_error = stacked.min(axis=1)
        minimum_mask = np.isclose(stacked, minimum_error[:, None], rtol=0.0, atol=1e-12)
        unique_minimum = minimum_mask.sum(axis=1) == 1
        model_order = list(absolute_errors)
        record = {
            "scheme_id": scheme_id,
            "n_observations": len(wide),
            "sequence_graph_residual_pearson": residual_pearson,
            "sequence_graph_residual_spearman": residual_spearman,
            "oracle_is_diagnostic_only": True,
            "oracle_mae": _metric_values(truth, oracle)["mae"],
            "oracle_rmse": _metric_values(truth, oracle)["rmse"],
        }
        for index, model in enumerate(model_order):
            record[f"{model}__unique_best_fraction"] = float(
                np.mean(minimum_mask[:, index] & unique_minimum)
            )
        rows.append(record)
    return pd.DataFrame(rows)


def inspect_sequence_structure_complementarity_inputs(
    repo_root: Path,
) -> Mapping[str, object]:
    repo_root = repo_root.resolve()
    _verify_guards(repo_root)
    sequence_settings = _selected_sequence_settings(repo_root)
    graph_epochs = _selected_graph_epochs(repo_root)
    base = _final_reference_predictions(repo_root)
    return {
        "status": "PASS",
        "analysis_scope": "sequence_structure_predictive_complementarity",
        "analysis_role": "secondary_sequence_structure_analysis",
        "model_id": MODEL_ID,
        "candidate_graph_weights": list(GRAPH_WEIGHTS),
        "selection_measure": "validation_mae",
        "selection_population": (
            "inner validation rows for nested schemes; "
            "five source-protein grouped folds "
            "for cross-protein schemes"
        ),
        "final_evaluation_labels_used_for_weight_selection": False,
        "cross_protein_target_labels_used_for_weight_selection": False,
        "existing_sequence_setting_count": len(sequence_settings),
        "existing_graph_epoch_count": len(graph_epochs),
        "matched_final_prediction_count": len(base),
        "validation_graph_training_seed_count": len(graph_stage.FINAL_REPEAT_SEEDS),
        "source_guards": dict(SOURCE_GUARDS),
        "result_guards": dict(RESULT_GUARDS),
    }


def run_sequence_structure_complementarity(
    repo_root: Path,
    *,
    device_name: str = "cuda",
    batch_size: int = graph_stage.BATCH_SIZE,
) -> Mapping[str, object]:
    repo_root = repo_root.resolve()
    audit = inspect_sequence_structure_complementarity_inputs(repo_root)
    graph_stage._configure_deterministic_execution()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SequenceStructureComplementarityError("CUDA is not available")
    device = torch.device(device_name)
    tables = graph_stage._load_tables(repo_root)
    graph_stage._validate_input_contract(tables)
    observations = graph_stage._build_analysis_observations(tables)
    contexts = graph_stage._make_contexts(tables, observations)
    embedding_index, mutant_embeddings, wild_type_embeddings = (
        combined_stage._load_embedding_data(repo_root)
    )
    sequence_settings = _selected_sequence_settings(repo_root)
    graph_epochs = _selected_graph_epochs(repo_root)
    output_dir = repo_root / "data/interim" / OUTPUT_DIR_NAME
    checkpoint_dir = output_dir / VALIDATION_CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    validation_frames = []
    selection_frames = []
    selected_weights: dict[tuple[str, str], float] = {}

    for context in contexts:
        scheme_id = str(context["scheme_id"])
        iteration = str(context["evaluation_iteration"])
        group_column = str(context["group_column"])
        condition_mode = graph_stage.CONDITION_MODE_BY_SCHEME[scheme_id]
        setting = _lookup_setting(sequence_settings, scheme_id, iteration)
        epoch_row = _lookup_setting(graph_epochs, scheme_id, iteration)
        epoch = int(epoch_row["selected_epoch"])

        if "source_folds" not in context:
            path = _checkpoint_path(output_dir, scheme_id, f"{iteration}__validation")
            if path.is_file():
                dev = _read_csv(path)
                _validate_validation_checkpoint(
                    dev,
                    context["validation"],
                    scheme_id,
                    iteration,
                    "inner_validation",
                    group_column,
                    setting,
                    epoch,
                )
            else:
                seq = _sequence_prediction(
                    context["train"],
                    context["validation"],
                    setting,
                    embedding_index,
                    mutant_embeddings,
                    wild_type_embeddings,
                    condition_mode,
                )
                graph = _graph_prediction(
                    context["train"],
                    context["validation"],
                    epoch,
                    tables,
                    condition_mode,
                    device,
                    f"{scheme_id} {iteration} validation graph",
                    batch_size,
                )
                dev = _validation_frame(
                    scheme_id,
                    iteration,
                    "inner_validation",
                    context["validation"],
                    group_column,
                    seq,
                    graph,
                    setting,
                    epoch,
                )
                _write_csv(dev, path)
            selection, graph_weight = _select_weight(dev, f"{scheme_id}/{iteration}")
            selection.insert(0, "scheme_id", scheme_id)
            selection.insert(1, "evaluation_iteration", iteration)
            selection_frames.append(selection)
            selected_weights[(scheme_id, iteration)] = graph_weight
            validation_frames.append(dev)
            continue

        fold_frames = []
        for fold_record in context["source_folds"]:
            fold = int(fold_record["fold"])
            path = _checkpoint_path(output_dir, scheme_id, f"source_cv_{fold}")
            if path.is_file():
                dev = _read_csv(path)
                _validate_validation_checkpoint(
                    dev,
                    fold_record["validation"],
                    scheme_id,
                    "target",
                    f"source_cv_{fold}",
                    group_column,
                    setting,
                    epoch,
                )
            else:
                seq = _sequence_prediction(
                    fold_record["train"],
                    fold_record["validation"],
                    setting,
                    embedding_index,
                    mutant_embeddings,
                    wild_type_embeddings,
                    condition_mode,
                )
                graph = _graph_prediction(
                    fold_record["train"],
                    fold_record["validation"],
                    epoch,
                    tables,
                    condition_mode,
                    device,
                    f"{scheme_id} source-CV {fold} validation graph",
                    batch_size,
                )
                dev = _validation_frame(
                    scheme_id,
                    "target",
                    f"source_cv_{fold}",
                    fold_record["validation"],
                    group_column,
                    seq,
                    graph,
                    setting,
                    epoch,
                )
                _write_csv(dev, path)
            fold_frames.append(dev)
            validation_frames.append(dev)
        source_validation = pd.concat(fold_frames, ignore_index=True)
        if source_validation["observation_id"].duplicated().any():
            raise SequenceStructureComplementarityError(
                f"Source-CV validation observations are duplicated for {scheme_id}"
            )
        selection, graph_weight = _select_weight(
            source_validation, f"{scheme_id}/source_cv"
        )
        selection.insert(0, "scheme_id", scheme_id)
        selection.insert(1, "evaluation_iteration", "target")
        selection_frames.append(selection)
        selected_weights[(scheme_id, "target")] = graph_weight

    validation = pd.concat(validation_frames, ignore_index=True)
    selection = pd.concat(selection_frames, ignore_index=True)
    final = _final_reference_predictions(repo_root)
    weight_keys = pd.DataFrame(
        [
            {
                "scheme_id": scheme,
                "evaluation_iteration": iteration,
                "graph_weight": weight,
            }
            for (scheme, iteration), weight in selected_weights.items()
        ]
    )
    final = final.merge(
        weight_keys,
        on=["scheme_id", "evaluation_iteration"],
        how="left",
        validate="many_to_one",
    )
    if final["graph_weight"].isna().any():
        raise SequenceStructureComplementarityError(
            "One or more final rows lack a selected weight"
        )
    final["sequence_weight"] = 1.0 - final["graph_weight"]
    final["y_pred_selected"] = (
        final["graph_weight"] * final["y_pred"]
        + final["sequence_weight"] * final["y_pred_sequence"]
    )

    selected_prediction = final[
        [
            *graph_stage.COMPARISON_KEY,
            *graph_stage.MATCHED_FIELDS,
            "graph_weight",
            "sequence_weight",
        ]
    ].copy()
    selected_prediction["model_id"] = MODEL_ID
    selected_prediction["y_pred"] = final["y_pred_selected"].to_numpy(dtype=float)

    references = []
    for model_id, column in (
        (GRAPH_MODEL, "y_pred"),
        (SEQUENCE_MODEL, "y_pred_sequence"),
        (COMBINED_MODEL, "y_pred_combined"),
    ):
        frame = final[[*graph_stage.COMPARISON_KEY, *graph_stage.MATCHED_FIELDS]].copy()
        frame["model_id"] = model_id
        frame["y_pred"] = final[column].to_numpy(dtype=float)
        references.append(frame)
    all_predictions = pd.concat([selected_prediction, *references], ignore_index=True)
    if all_predictions.duplicated([*graph_stage.COMPARISON_KEY, "model_id"]).any():
        raise SequenceStructureComplementarityError(
            "Final prediction identities are duplicated"
        )

    metrics = _metrics(all_predictions)
    paired = _paired_comparisons(all_predictions)
    diagnostics = _diagnostics(all_predictions)

    _write_csv(validation, output_dir / "sequence_structure_validation_predictions.csv")
    _write_csv(selection, output_dir / "sequence_structure_model_selection.csv")
    _write_csv(selected_prediction, output_dir / "sequence_structure_predictions.csv")
    _write_csv(metrics, output_dir / "sequence_structure_metrics.csv")
    _write_csv(paired, output_dir / "sequence_structure_paired_comparisons.csv")
    _write_csv(diagnostics, output_dir / "sequence_structure_complementarity.csv")

    summary = {
        **audit,
        "status": "PASS",
        "device": device_name,
        "validation_prediction_count": len(validation),
        "final_prediction_count": len(selected_prediction),
        "metric_row_count": len(metrics),
        "paired_comparison_row_count": len(paired),
        "selected_graph_weights": {
            f"{scheme}/{iteration}": weight
            for (scheme, iteration), weight in sorted(selected_weights.items())
        },
        "final_evaluation_labels_used_for_weight_selection": False,
        "cross_protein_target_labels_used_for_weight_selection": False,
        "diagnostic_oracle_used_for_model_selection": False,
        "existing_sequence_predictions_changed": False,
        "existing_graph_eval_predictions_changed": False,
        "existing_sequence_residue_graph_eval_predictions_changed": False,
        "outputs": [
            "sequence_structure_validation_predictions.csv",
            "sequence_structure_model_selection.csv",
            "sequence_structure_predictions.csv",
            "sequence_structure_metrics.csv",
            "sequence_structure_paired_comparisons.csv",
            "sequence_structure_complementarity.csv",
            "sequence_structure_summary.json",
        ],
    }
    (output_dir / "sequence_structure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
