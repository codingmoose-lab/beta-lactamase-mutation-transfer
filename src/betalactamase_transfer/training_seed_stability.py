from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from betalactamase_transfer import residue_geometry_neural_evaluation as stage33
from betalactamase_transfer import residue_graph_neural_evaluation as stage32
from betalactamase_transfer import sequence_residue_graph_neural_evaluation as stage34


class TrainingSeedStabilityError(RuntimeError):
    pass


OUTPUT_DIR_NAME = "training_seed_stability"
CHECKPOINT_DIR_NAME = "seed_checkpoints"
ADDITIONAL_PREDICTION_FILE = "additional_training_seed_predictions.csv.gz"
ADDITIONAL_HISTORY_FILE = "additional_training_seed_history.csv"
SEED_METRIC_FILE = "training_seed_metrics.csv"
STABILITY_FILE = "training_seed_metric_stability.csv"
COMPARISON_FILE = "training_seed_model_comparisons.csv"
COMPARISON_SUMMARY_FILE = "training_seed_comparison_summary.csv"
SUMMARY_FILE = "training_seed_stability_summary.json"
TRAINING_COMPLETE_FILE = "training_complete.json"

TOTAL_SEED_COUNT = 10
EXISTING_SEED_COUNT = 3
ADDITIONAL_SEED_COUNT = TOTAL_SEED_COUNT - EXISTING_SEED_COUNT

GRAPH_SEEDS = tuple(range(32101, 32101 + TOTAL_SEED_COUNT))
GEOMETRY_SEEDS = tuple(range(33101, 33101 + TOTAL_SEED_COUNT))
JOINT_SEEDS = GRAPH_SEEDS

MODEL_CENTRE = stage32.CENTRE_MODEL
MODEL_GRAPH = stage32.GRAPH_MODEL
MODEL_GEOMETRY = stage33.MODEL_ID
MODEL_JOINT = stage34.MODEL_ID
MODEL_IDS = (MODEL_CENTRE, MODEL_GRAPH, MODEL_GEOMETRY, MODEL_JOINT)

SEEDS_BY_MODEL = {
    MODEL_CENTRE: GRAPH_SEEDS,
    MODEL_GRAPH: GRAPH_SEEDS,
    MODEL_GEOMETRY: GEOMETRY_SEEDS,
    MODEL_JOINT: JOINT_SEEDS,
}

CORE_SCHEME_IDS = tuple(stage32.CORE_SCHEME_IDS)
TARGET_COLUMN = stage32.TARGET_COLUMN
COMPARISON_KEY = tuple(stage32.COMPARISON_KEY)
MATCHED_FIELDS = tuple(stage32.MATCHED_FIELDS)

PRIMARY_COMPARISONS = (
    (MODEL_GRAPH, MODEL_CENTRE, "matched_training_seed"),
    (MODEL_GEOMETRY, MODEL_GRAPH, "all_seed_combinations"),
    (MODEL_JOINT, MODEL_GRAPH, "matched_training_seed"),
)

METRIC_NAMES = ("spearman", "pearson", "mae", "rmse")
TARGET_ABSOLUTE_TOLERANCE = 1e-12

EXPECTED_MODEL_PREDICTION_COUNT = 118204
EXPECTED_CONTEXT_COUNT = 17
EXPECTED_EXISTING_ROWS = {
    MODEL_CENTRE: EXPECTED_MODEL_PREDICTION_COUNT * EXISTING_SEED_COUNT,
    MODEL_GRAPH: EXPECTED_MODEL_PREDICTION_COUNT * EXISTING_SEED_COUNT,
    MODEL_GEOMETRY: EXPECTED_MODEL_PREDICTION_COUNT * EXISTING_SEED_COUNT,
    MODEL_JOINT: EXPECTED_MODEL_PREDICTION_COUNT * EXISTING_SEED_COUNT,
}
EXPECTED_ADDITIONAL_ROWS = (
    EXPECTED_MODEL_PREDICTION_COUNT * len(MODEL_IDS) * ADDITIONAL_SEED_COUNT
)

SOURCE_HASHES = {
    "src/betalactamase_transfer/residue_graph_neural_evaluation.py": (
        "b66d1c2873f0ae6d62aa3ddfddc24e4a6f761fc9dc9d8f8edf8d521d509e3154"
    ),
    "src/betalactamase_transfer/residue_geometry_neural_evaluation.py": (
        "ead8c636124906c443c22db9e3b894ec10fccb1c62bdcd89f2274988fdac90a3"
    ),
    "src/betalactamase_transfer/sequence_residue_graph_neural_evaluation.py": (
        "b408ea90b696ccfd6e84d779018681c27556fec6f1d86e51945d0a50d80d1edf"
    ),
}

RESULT_HASHES = {
    (
        "data/interim/residue_graph_neural_evaluation/"
        "residue_graph_neural_predictions.csv"
    ): ("98aa268232824720aecd2a8510d4169e4faba4f4dfd71c31659b92088af0203a"),
    (
        "data/interim/residue_graph_neural_evaluation/"
        "residue_graph_neural_repeat_predictions.csv"
    ): ("a8464a9cb4f3153a6d190b63a4fa62bf96370facf2cd7b561b15e4963c8d5367"),
    (
        "data/interim/residue_graph_neural_evaluation/"
        "residue_graph_neural_training_history.csv"
    ): ("7b5ca9d0bb055636c21a44dc6576e94b951ab9876ab54d733d81b34025ed60e4"),
    (
        "data/interim/residue_geometry_neural_evaluation/"
        "residue_geometry_neural_predictions.csv"
    ): ("478008269ed73f584ca7c5677f747cb97c08547db4d5c8925f82a2aac3243ea6"),
    (
        "data/interim/residue_geometry_neural_evaluation/"
        "residue_geometry_neural_repeat_predictions.csv"
    ): ("bbcdcad02b58dd0c98c76924fb4d5336b17dba8060a93b8755a69e0066e5638e"),
    (
        "data/interim/sequence_residue_graph_neural_evaluation/"
        "sequence_residue_graph_predictions.csv"
    ): ("6d6ddae3f9654b8dd0a79dd750ede8755b2ac35bb3f8454d0e4d3457796fed80"),
    (
        "data/interim/sequence_residue_graph_neural_evaluation/"
        "sequence_residue_graph_repeat_predictions.csv"
    ): ("a6e54ae0980338e554c79b05cda6f5b96b905d50f7b51a46c6bd6c252c850a9c"),
    (
        "data/interim/sequence_residue_graph_neural_evaluation/"
        "sequence_residue_graph_training_history.csv"
    ): ("e9e4c59b7f36b6dd185fd36390fb967f42a665bfa69dd37a208b96bbb4a6f986"),
    (
        "data/interim/biological_model_difference_analysis/"
        "biological_model_difference_summary.json"
    ): ("81323e6852c4ab4f19dbe4052955cb766864319d8ce2ccb36af7aef5b4868fc6"),
}


PREDICTION_COLUMNS = (
    *COMPARISON_KEY,
    *MATCHED_FIELDS,
    "model_id",
    "seed_set_index",
    "training_seed",
    "record_source",
    "y_pred",
)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def _write_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    return stage32._metric_values(y_true, y_pred)


def _metric_difference(metric: str, value_a: float, value_b: float) -> float:
    if metric in {"mae", "rmse"}:
        return float(value_b - value_a)
    if metric in {"spearman", "pearson"}:
        return float(value_a - value_b)
    raise TrainingSeedStabilityError(f"Unknown metric: {metric}")


def _seed_plan() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        seeds = SEEDS_BY_MODEL[model_id]
        for index, seed in enumerate(seeds, start=1):
            rows.append(
                {
                    "model_id": model_id,
                    "seed_set_index": index,
                    "training_seed": seed,
                    "record_source": "existing_stage_record"
                    if index <= EXISTING_SEED_COUNT
                    else "stage37_additional_fit",
                }
            )
    return pd.DataFrame(rows)


def _verify_hashes(repo_root: Path) -> None:
    for relative, expected in {**SOURCE_HASHES, **RESULT_HASHES}.items():
        path = repo_root / relative
        if not path.is_file():
            raise TrainingSeedStabilityError(f"Required file is missing: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise TrainingSeedStabilityError(
                f"File identity differs for {relative}: {observed} != {expected}"
            )


def _load_history(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"evaluation_iteration": str}, low_memory=False)


def _selected_epochs_from_history(
    history: pd.DataFrame,
    model_ids: Sequence[str],
) -> dict[tuple[str, str, str], int]:
    required = {"scheme_id", "evaluation_iteration", "model_id", "fit_role", "epoch"}
    missing = required.difference(history.columns)
    if missing:
        raise TrainingSeedStabilityError(
            f"Training history lacks required columns: {sorted(missing)}"
        )
    selected = history[
        history["fit_role"].astype(str).eq("selected_epoch")
        & history["model_id"].astype(str).isin(tuple(model_ids))
    ].copy()
    selected["evaluation_iteration"] = selected["evaluation_iteration"].astype(str)
    identity = ["scheme_id", "evaluation_iteration", "model_id"]
    if selected.duplicated(identity).any():
        raise TrainingSeedStabilityError("Selected-epoch records are duplicated")
    result: dict[tuple[str, str, str], int] = {}
    for row in selected.itertuples(index=False):
        epoch = int(row.epoch)
        if epoch < 1:
            raise TrainingSeedStabilityError("Selected epoch must be positive")
        result[
            (str(row.scheme_id), str(row.evaluation_iteration), str(row.model_id))
        ] = epoch
    return result


def _selected_epoch_contract(repo_root: Path) -> dict[tuple[str, str, str], int]:
    graph_history = _load_history(
        repo_root
        / (
            "data/interim/residue_graph_neural_evaluation/"
            "residue_graph_neural_training_history.csv"
        )
    )
    geometry_history = _load_history(
        repo_root
        / (
            "data/interim/residue_geometry_neural_evaluation/"
            "residue_geometry_neural_training_history.csv"
        )
    )
    joint_history = _load_history(
        repo_root
        / (
            "data/interim/sequence_residue_graph_neural_evaluation/"
            "sequence_residue_graph_training_history.csv"
        )
    )
    selected: dict[tuple[str, str, str], int] = {}
    for frame, models in (
        (graph_history, (MODEL_CENTRE, MODEL_GRAPH)),
        (geometry_history, (MODEL_GEOMETRY,)),
        (joint_history, (MODEL_JOINT,)),
    ):
        current = _selected_epochs_from_history(frame, models)
        overlap = set(selected).intersection(current)
        if overlap:
            raise TrainingSeedStabilityError(
                f"Selected-epoch keys overlap: {sorted(overlap)}"
            )
        selected.update(current)
    expected = EXPECTED_CONTEXT_COUNT * len(MODEL_IDS)
    if len(selected) != expected:
        raise TrainingSeedStabilityError(
            f"Expected {expected} selected model/context epochs, "
            f"observed {len(selected)}"
        )
    return selected


def _existing_seed_file(repo_root: Path, model_id: str) -> Path:
    if model_id in {MODEL_CENTRE, MODEL_GRAPH}:
        return (
            repo_root
            / "data/interim/residue_graph_neural_evaluation"
            / "residue_graph_neural_repeat_predictions.csv"
        )
    if model_id == MODEL_GEOMETRY:
        return (
            repo_root
            / "data/interim/residue_geometry_neural_evaluation"
            / "residue_geometry_neural_repeat_predictions.csv"
        )
    if model_id == MODEL_JOINT:
        return (
            repo_root
            / "data/interim/sequence_residue_graph_neural_evaluation"
            / "sequence_residue_graph_repeat_predictions.csv"
        )
    raise TrainingSeedStabilityError(f"Unknown model: {model_id}")


def _normalise_existing_seed_predictions(
    repo_root: Path, model_id: str
) -> pd.DataFrame:
    frame = pd.read_csv(
        _existing_seed_file(repo_root, model_id),
        dtype={"evaluation_iteration": str, "observation_id": str},
        low_memory=False,
    )
    if "training_repeat" not in frame.columns:
        raise TrainingSeedStabilityError(
            f"Existing prediction file lacks its training-run index for {model_id}"
        )
    if "model_id" in frame.columns:
        frame = frame[frame["model_id"].astype(str).eq(model_id)].copy()
    required = {
        *COMPARISON_KEY,
        *MATCHED_FIELDS,
        "model_id",
        "training_repeat",
        "y_pred",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise TrainingSeedStabilityError(
            f"Existing predictions lack required columns for {model_id}: "
            f"{sorted(missing)}"
        )
    expected_rows = EXPECTED_EXISTING_ROWS[model_id]
    if len(frame) != expected_rows:
        raise TrainingSeedStabilityError(
            f"Unexpected existing prediction count for {model_id}: {len(frame)}"
        )
    run_index = pd.to_numeric(frame["training_repeat"], errors="raise").astype(int)
    if set(run_index.unique()) != set(range(1, EXISTING_SEED_COUNT + 1)):
        raise TrainingSeedStabilityError(
            f"Existing training-run indices differ for {model_id}"
        )
    frame["seed_set_index"] = run_index
    seeds = SEEDS_BY_MODEL[model_id]
    frame["training_seed"] = frame["seed_set_index"].map(
        {index: seeds[index - 1] for index in range(1, EXISTING_SEED_COUNT + 1)}
    )
    frame["record_source"] = "existing_stage_record"
    output = frame[[*PREDICTION_COLUMNS]].copy()
    identity = [*COMPARISON_KEY, "model_id", "training_seed"]
    if output.duplicated(identity).any():
        raise TrainingSeedStabilityError(
            f"Existing seed predictions contain duplicate identities for {model_id}"
        )
    if not np.isfinite(output["y_true"].to_numpy(dtype=float)).all():
        raise TrainingSeedStabilityError(
            f"Existing outcomes are not finite for {model_id}"
        )
    if not np.isfinite(output["y_pred"].to_numpy(dtype=float)).all():
        raise TrainingSeedStabilityError(
            f"Existing predictions are not finite for {model_id}"
        )
    return output


def _context_fitting_and_test(
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        raise TrainingSeedStabilityError("Unable to order evaluation rows")
    return fitting_rows, ordered_test


def _prediction_frame(
    context: Mapping[str, object],
    model_id: str,
    seed_set_index: int,
    training_seed: int,
    prediction: np.ndarray,
) -> pd.DataFrame:
    reference = context["reference"].reset_index(drop=True)
    if len(reference) != len(prediction):
        raise TrainingSeedStabilityError(
            "Prediction length differs from evaluation rows"
        )
    frame = reference[[*COMPARISON_KEY, *MATCHED_FIELDS]].copy()
    frame["model_id"] = model_id
    frame["seed_set_index"] = int(seed_set_index)
    frame["training_seed"] = int(training_seed)
    frame["record_source"] = "stage37_additional_fit"
    frame["y_pred"] = np.asarray(prediction, dtype=float)
    return frame[[*PREDICTION_COLUMNS]]


def _checkpoint_token(model_id: str, context: Mapping[str, object]) -> str:
    scheme = str(context["scheme_id"])
    iteration = str(context["evaluation_iteration"]).replace("/", "_")
    return f"{model_id}__{scheme}__{iteration}"


def _checkpoint_paths(
    output_dir: Path, model_id: str, context: Mapping[str, object]
) -> tuple[Path, Path]:
    token = _checkpoint_token(model_id, context)
    base = output_dir / CHECKPOINT_DIR_NAME
    return base / f"{token}.csv.gz", base / f"{token}__history.csv"


def _validate_checkpoint(
    frame: pd.DataFrame,
    context: Mapping[str, object],
    model_id: str,
) -> None:
    if frame.empty:
        return
    required = set(PREDICTION_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise TrainingSeedStabilityError(
            f"Seed checkpoint lacks columns: {sorted(missing)}"
        )
    if set(frame["model_id"].astype(str)) != {model_id}:
        raise TrainingSeedStabilityError("Seed checkpoint model differs")
    expected_seeds = set(SEEDS_BY_MODEL[model_id][EXISTING_SEED_COUNT:])
    observed_seeds = set(
        pd.to_numeric(frame["training_seed"], errors="raise").astype(int)
    )
    if not observed_seeds.issubset(expected_seeds):
        raise TrainingSeedStabilityError("Seed checkpoint contains an unplanned seed")
    reference_ids = context["reference"]["observation_id"].astype(str).tolist()
    for seed, part in frame.groupby("training_seed", sort=False):
        if part["observation_id"].astype(str).tolist() != reference_ids:
            raise TrainingSeedStabilityError(
                f"Seed checkpoint evaluation rows differ for seed {seed}"
            )
    identity = [*COMPARISON_KEY, "model_id", "training_seed"]
    if frame.duplicated(identity).any():
        raise TrainingSeedStabilityError(
            "Seed checkpoint contains duplicate identities"
        )


def _load_checkpoint(
    output_dir: Path, model_id: str, context: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_path, history_path = _checkpoint_paths(output_dir, model_id, context)
    if not prediction_path.is_file():
        return pd.DataFrame(columns=PREDICTION_COLUMNS), pd.DataFrame()
    frame = pd.read_csv(
        prediction_path,
        dtype={"evaluation_iteration": str, "observation_id": str},
        low_memory=False,
    )
    _validate_checkpoint(frame, context, model_id)
    history = (
        pd.read_csv(history_path, dtype={"evaluation_iteration": str}, low_memory=False)
        if history_path.is_file()
        else pd.DataFrame()
    )
    return frame, history


def _save_checkpoint(
    output_dir: Path,
    model_id: str,
    context: Mapping[str, object],
    predictions: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    prediction_path, history_path = _checkpoint_paths(output_dir, model_id, context)
    _validate_checkpoint(predictions, context, model_id)
    prediction_temp = prediction_path.with_name(prediction_path.name + ".tmp")
    history_temp = history_path.with_name(history_path.name + ".tmp")
    _write_csv_gzip(predictions, prediction_temp)
    _write_csv(history, history_temp)
    prediction_temp.replace(prediction_path)
    history_temp.replace(history_path)


def _fit_one_seed(
    model_id: str,
    context: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    embedding_data: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None,
    device: torch.device,
    training_seed: int,
    selected_epoch: int,
    batch_size: int,
    seed_set_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fitting_rows, ordered_test = _context_fitting_and_test(context)
    scheme_id = str(context["scheme_id"])
    iteration = str(context["evaluation_iteration"])
    condition_mode = stage32.CONDITION_MODE_BY_SCHEME[scheme_id]
    description = (
        f"{scheme_id} {iteration} {model_id} training seed "
        f"{seed_set_index}/{TOTAL_SEED_COUNT}"
    )

    if model_id in {MODEL_CENTRE, MODEL_GRAPH}:
        prediction, history = stage32._fit_fixed_epochs_and_predict(
            model_id,
            fitting_rows,
            ordered_test,
            tables["nodes"],
            tables["edges"],
            tables["residue_features"],
            condition_mode,
            device,
            training_seed,
            selected_epoch,
            description,
            batch_size,
        )
    elif model_id == MODEL_GEOMETRY:
        prediction, history = stage33._fit_fixed_epochs_and_predict(
            fitting_rows,
            ordered_test,
            tables["nodes"],
            tables["edges"],
            tables["residue_features"],
            condition_mode,
            device,
            training_seed,
            selected_epoch,
            description,
            batch_size,
        )
    elif model_id == MODEL_JOINT:
        if embedding_data is None:
            raise TrainingSeedStabilityError("Sequence embedding data are unavailable")
        embedding_index, mutant_embeddings, wild_type_embeddings = embedding_data
        prediction, history = stage34._fit_fixed_epochs_and_predict(
            fitting_rows,
            ordered_test,
            tables,
            embedding_index,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode,
            device,
            training_seed,
            selected_epoch,
            description,
            batch_size,
        )
    else:
        raise TrainingSeedStabilityError(f"Unknown model: {model_id}")

    prediction_frame = _prediction_frame(
        context,
        model_id,
        seed_set_index,
        training_seed,
        prediction,
    )
    history_frame = pd.DataFrame(
        [
            {
                "scheme_id": scheme_id,
                "evaluation_iteration": iteration,
                "model_id": model_id,
                "seed_set_index": seed_set_index,
                "training_seed": training_seed,
                "selected_epoch_count": selected_epoch,
                **row,
            }
            for row in history
        ]
    )
    return prediction_frame, history_frame


def _count_prediction_rows(path: Path) -> int:
    rows = 0
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        rows += len(chunk)
    return rows


def _completed_additional_paths(output_dir: Path) -> tuple[Path, Path] | None:
    prediction_path = output_dir / ADDITIONAL_PREDICTION_FILE
    history_path = output_dir / ADDITIONAL_HISTORY_FILE
    if not prediction_path.is_file() or not history_path.is_file():
        return None
    row_count = _count_prediction_rows(prediction_path)
    if row_count != EXPECTED_ADDITIONAL_ROWS:
        raise TrainingSeedStabilityError(
            f"Unexpected additional prediction count: {row_count}"
        )
    return prediction_path, history_path


def _combine_checkpoint_files(output_dir: Path) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / CHECKPOINT_DIR_NAME
    prediction_files = sorted(
        path
        for path in checkpoint_dir.glob("*.csv.gz")
        if not path.name.endswith("__history.csv.gz")
    )
    history_files = sorted(checkpoint_dir.glob("*__history.csv"))
    expected_files = len(MODEL_IDS) * EXPECTED_CONTEXT_COUNT
    if len(prediction_files) != expected_files or len(history_files) != expected_files:
        raise TrainingSeedStabilityError(
            "Completed seed checkpoints do not cover every model/context"
        )

    prediction_path = output_dir / ADDITIONAL_PREDICTION_FILE
    history_path = output_dir / ADDITIONAL_HISTORY_FILE
    prediction_temp = prediction_path.with_name(prediction_path.name + ".tmp")
    history_temp = history_path.with_name(history_path.name + ".tmp")

    with gzip.open(prediction_temp, "wt", encoding="utf-8", newline="") as destination:
        wrote_header = False
        for source_path in prediction_files:
            with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header:
                    raise TrainingSeedStabilityError(
                        f"Seed checkpoint is empty: {source_path.name}"
                    )
                if not wrote_header:
                    destination.write(header)
                    wrote_header = True
                shutil.copyfileobj(source, destination)

    with history_temp.open("w", encoding="utf-8", newline="") as destination:
        wrote_header = False
        for source_path in history_files:
            with source_path.open("r", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header:
                    raise TrainingSeedStabilityError(
                        f"Seed history checkpoint is empty: {source_path.name}"
                    )
                if not wrote_header:
                    destination.write(header)
                    wrote_header = True
                shutil.copyfileobj(source, destination)

    prediction_temp.replace(prediction_path)
    history_temp.replace(history_path)
    row_count = _count_prediction_rows(prediction_path)
    if row_count != EXPECTED_ADDITIONAL_ROWS:
        raise TrainingSeedStabilityError(
            f"Unexpected additional prediction count after combination: {row_count}"
        )
    shutil.rmtree(checkpoint_dir)
    return prediction_path, history_path


def _fit_additional_seeds(
    repo_root: Path,
    output_dir: Path,
    contexts: Sequence[Mapping[str, object]],
    tables: Mapping[str, pd.DataFrame],
    selected_epochs: Mapping[tuple[str, str, str], int],
    device: torch.device,
    batch_size: int,
) -> tuple[Path, Path]:
    saved = _completed_additional_paths(output_dir)
    if saved is not None:
        return saved

    embedding_data = stage34._load_embedding_data(repo_root)
    for model_id in MODEL_IDS:
        seeds = SEEDS_BY_MODEL[model_id]
        for context in contexts:
            scheme_id = str(context["scheme_id"])
            iteration = str(context["evaluation_iteration"])
            epoch_key = (scheme_id, iteration, model_id)
            if epoch_key not in selected_epochs:
                raise TrainingSeedStabilityError(
                    f"Selected epoch is missing for {epoch_key}"
                )
            selected_epoch = int(selected_epochs[epoch_key])
            context_predictions, context_history = _load_checkpoint(
                output_dir, model_id, context
            )
            completed = set(
                pd.to_numeric(
                    context_predictions.get("training_seed", pd.Series(dtype=int)),
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )
            for seed_set_index in range(EXISTING_SEED_COUNT + 1, TOTAL_SEED_COUNT + 1):
                training_seed = int(seeds[seed_set_index - 1])
                if training_seed in completed:
                    continue
                new_predictions, new_history = _fit_one_seed(
                    model_id,
                    context,
                    tables,
                    embedding_data if model_id == MODEL_JOINT else None,
                    device,
                    training_seed,
                    selected_epoch,
                    batch_size,
                    seed_set_index,
                )
                context_predictions = pd.concat(
                    [context_predictions, new_predictions], ignore_index=True
                )
                context_history = pd.concat(
                    [context_history, new_history], ignore_index=True
                )
                _save_checkpoint(
                    output_dir,
                    model_id,
                    context,
                    context_predictions,
                    context_history,
                )
                completed.add(training_seed)

            expected = set(seeds[EXISTING_SEED_COUNT:])
            if completed != expected:
                raise TrainingSeedStabilityError(
                    f"Additional seeds are incomplete for {epoch_key}: "
                    f"{sorted(completed)}"
                )

    return _combine_checkpoint_files(output_dir)


def _metric_accumulator() -> dict[tuple[str, str, int, int], dict[str, object]]:
    return {}


def _accumulate_metric_rows(
    accumulator: dict[tuple[str, str, int, int], dict[str, object]],
    frame: pd.DataFrame,
) -> None:
    required = {
        "scheme_id",
        "model_id",
        "seed_set_index",
        "training_seed",
        "evaluation_group_id",
        "y_true",
        "y_pred",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise TrainingSeedStabilityError(
            f"Training-seed prediction records lack columns: {sorted(missing)}"
        )
    for key, part in frame.groupby(
        ["scheme_id", "model_id", "seed_set_index", "training_seed"],
        sort=False,
    ):
        scheme_id, model_id, seed_set_index, training_seed = key
        normalized_key = (
            str(scheme_id),
            str(model_id),
            int(seed_set_index),
            int(training_seed),
        )
        record = accumulator.setdefault(
            normalized_key,
            {"y_true": [], "y_pred": [], "groups": set()},
        )
        y_true = part["y_true"].to_numpy(dtype=float)
        y_pred = part["y_pred"].to_numpy(dtype=float)
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise TrainingSeedStabilityError("Training-seed predictions are not finite")
        record["y_true"].append(y_true)
        record["y_pred"].append(y_pred)
        record["groups"].update(part["evaluation_group_id"].astype(str).tolist())


def _accumulate_existing_model(
    repo_root: Path,
    model_id: str,
    accumulator: dict[tuple[str, str, int, int], dict[str, object]],
) -> None:
    path = _existing_seed_file(repo_root, model_id)
    seeds = SEEDS_BY_MODEL[model_id]
    for chunk in pd.read_csv(
        path,
        dtype={"evaluation_iteration": str, "observation_id": str},
        chunksize=100_000,
        low_memory=False,
    ):
        if "model_id" in chunk.columns:
            chunk = chunk[chunk["model_id"].astype(str).eq(model_id)].copy()
        if chunk.empty:
            continue
        run_index = pd.to_numeric(chunk["training_repeat"], errors="raise").astype(int)
        chunk["seed_set_index"] = run_index
        chunk["training_seed"] = run_index.map(
            {index: seeds[index - 1] for index in range(1, EXISTING_SEED_COUNT + 1)}
        )
        _accumulate_metric_rows(accumulator, chunk)


def _seed_metrics_from_sources(repo_root: Path, additional_path: Path) -> pd.DataFrame:
    accumulator = _metric_accumulator()
    for model_id in MODEL_IDS:
        _accumulate_existing_model(repo_root, model_id, accumulator)
    for chunk in pd.read_csv(
        additional_path,
        dtype={"evaluation_iteration": str, "observation_id": str},
        chunksize=100_000,
        low_memory=False,
    ):
        _accumulate_metric_rows(accumulator, chunk)

    rows: list[dict[str, object]] = []
    for (scheme_id, model_id, seed_index, training_seed), record in accumulator.items():
        y_true = np.concatenate(record["y_true"])
        y_pred = np.concatenate(record["y_pred"])
        metrics = _metric_values(y_true, y_pred)
        rows.append(
            {
                "scheme_id": scheme_id,
                "model_id": model_id,
                "seed_set_index": seed_index,
                "training_seed": training_seed,
                "n_observations": len(y_true),
                "n_groups": len(record["groups"]),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    expected = len(CORE_SCHEME_IDS) * len(MODEL_IDS) * TOTAL_SEED_COUNT
    if len(result) != expected:
        raise TrainingSeedStabilityError(
            f"Unexpected training-seed metric count: {len(result)}"
        )
    return result.sort_values(["scheme_id", "model_id", "seed_set_index"]).reset_index(
        drop=True
    )


def _stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme_id, model_id), part in metrics.groupby(["scheme_id", "model_id"]):
        for metric in METRIC_NAMES:
            values = part[metric].to_numpy(dtype=float)
            existing_values = part.loc[
                part["seed_set_index"].astype(int).le(EXISTING_SEED_COUNT), metric
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "model_id": model_id,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "existing_three_seed_mean": float(np.mean(existing_values)),
                    "ten_seed_mean_minus_existing_three_seed_mean": float(
                        np.mean(values) - np.mean(existing_values)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _comparison_rows(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_lookup = {
        (str(row.scheme_id), str(row.model_id), int(row.seed_set_index)): row
        for row in seed_metrics.itertuples(index=False)
    }
    for model_a, model_b, method in PRIMARY_COMPARISONS:
        for scheme_id in CORE_SCHEME_IDS:
            if method == "matched_training_seed":
                index_pairs = [
                    (index, index) for index in range(1, TOTAL_SEED_COUNT + 1)
                ]
            else:
                index_pairs = [
                    (index_a, index_b)
                    for index_a in range(1, TOTAL_SEED_COUNT + 1)
                    for index_b in range(1, TOTAL_SEED_COUNT + 1)
                ]
            for index_a, index_b in index_pairs:
                row_a = metric_lookup[(scheme_id, model_a, index_a)]
                row_b = metric_lookup[(scheme_id, model_b, index_b)]
                for metric in METRIC_NAMES:
                    value_a = float(getattr(row_a, metric))
                    value_b = float(getattr(row_b, metric))
                    difference = _metric_difference(metric, value_a, value_b)
                    rows.append(
                        {
                            "scheme_id": scheme_id,
                            "model_a": model_a,
                            "model_b": model_b,
                            "comparison_method": method,
                            "model_a_seed_set_index": index_a,
                            "model_b_seed_set_index": index_b,
                            "model_a_training_seed": int(row_a.training_seed),
                            "model_b_training_seed": int(row_b.training_seed),
                            "metric": metric,
                            "model_a_value": value_a,
                            "model_b_value": value_b,
                            "difference": difference,
                            "difference_interpretation": "positive_favours_model_a",
                        }
                    )
    return pd.DataFrame(rows)


def _comparison_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["scheme_id", "model_a", "model_b", "comparison_method", "metric"]
    for key, part in comparisons.groupby(group_columns, sort=False):
        scheme_id, model_a, model_b, method, metric = key
        differences = part["difference"].to_numpy(dtype=float)
        existing = part[
            part["model_a_seed_set_index"].astype(int).le(EXISTING_SEED_COUNT)
            & part["model_b_seed_set_index"].astype(int).le(EXISTING_SEED_COUNT)
        ]["difference"].to_numpy(dtype=float)
        tolerance = 1e-12
        expanded_mean = float(np.mean(differences))
        existing_mean = float(np.mean(existing))
        expanded_sign = (
            1 if expanded_mean > tolerance else -1 if expanded_mean < -tolerance else 0
        )
        existing_sign = (
            1 if existing_mean > tolerance else -1 if existing_mean < -tolerance else 0
        )
        rows.append(
            {
                "scheme_id": scheme_id,
                "model_a": model_a,
                "model_b": model_b,
                "comparison_method": method,
                "metric": metric,
                "comparison_count": len(part),
                "model_a_better_count": int(np.sum(differences > tolerance)),
                "model_b_better_count": int(np.sum(differences < -tolerance)),
                "tie_count": int(np.sum(np.abs(differences) <= tolerance)),
                "model_a_better_fraction": float(np.mean(differences > tolerance)),
                "mean_difference": expanded_mean,
                "median_difference": float(np.median(differences)),
                "minimum_difference": float(np.min(differences)),
                "maximum_difference": float(np.max(differences)),
                "existing_three_seed_mean_difference": existing_mean,
                "mean_direction_changed": bool(expanded_sign != existing_sign),
            }
        )
    return pd.DataFrame(rows)


def inspect_training_seed_stability_inputs(repo_root: Path) -> Mapping[str, object]:
    repo_root = repo_root.resolve()
    _verify_hashes(repo_root)
    if tuple(stage32.FINAL_REPEAT_SEEDS) != GRAPH_SEEDS[:EXISTING_SEED_COUNT]:
        raise TrainingSeedStabilityError("Stage-32 existing training seeds differ")
    if tuple(stage33.FINAL_REPEAT_SEEDS) != GEOMETRY_SEEDS[:EXISTING_SEED_COUNT]:
        raise TrainingSeedStabilityError("Stage-33 existing training seeds differ")
    if tuple(stage34.FINAL_REPEAT_SEEDS) != JOINT_SEEDS[:EXISTING_SEED_COUNT]:
        raise TrainingSeedStabilityError("Stage-34 existing training seeds differ")

    tables = stage32._load_tables(repo_root)
    observations = stage32._build_analysis_observations(tables)
    contexts = stage32._make_contexts(tables, observations)
    if len(contexts) != EXPECTED_CONTEXT_COUNT:
        raise TrainingSeedStabilityError(
            f"Expected {EXPECTED_CONTEXT_COUNT} contexts, observed {len(contexts)}"
        )
    selected_epochs = _selected_epoch_contract(repo_root)
    existing_counts: dict[str, int] = {}
    for model_id in MODEL_IDS:
        frame = _normalise_existing_seed_predictions(repo_root, model_id)
        existing_counts[model_id] = len(frame)

    stage36_summary = json.loads(
        (
            repo_root
            / "data/interim/biological_model_difference_analysis"
            / "biological_model_difference_summary.json"
        ).read_text(encoding="utf-8")
    )
    if stage36_summary.get("status") != "PASS":
        raise TrainingSeedStabilityError("Stage-36 summary does not report PASS")

    output_dir = repo_root / "data/interim" / OUTPUT_DIR_NAME
    return {
        "status": "PASS",
        "analysis_scope": "fixed_setting_training_seed_stability",
        "analysis_role": "secondary_training_variation_analysis",
        "total_training_seed_count": TOTAL_SEED_COUNT,
        "existing_training_seed_count": EXISTING_SEED_COUNT,
        "additional_training_seed_count": ADDITIONAL_SEED_COUNT,
        "model_ids": list(MODEL_IDS),
        "seed_sets": {model: list(seeds) for model, seeds in SEEDS_BY_MODEL.items()},
        "evaluation_context_count": len(contexts),
        "selected_epoch_record_count": len(selected_epochs),
        "existing_prediction_rows": existing_counts,
        "model_selection_performed": False,
        "validation_search_repeated": False,
        "evaluation_labels_used_for_setting_choice": False,
        "existing_primary_predictions_changed": False,
        "output_directory_exists": output_dir.exists(),
        "source_guards": dict(SOURCE_HASHES),
        "result_guards": dict(RESULT_HASHES),
    }


def _postprocess(
    repo_root: Path,
    output_dir: Path,
    additional_prediction_path: Path,
) -> Mapping[str, object]:
    seed_metrics = _seed_metrics_from_sources(repo_root, additional_prediction_path)
    stability = _stability_summary(seed_metrics)
    comparisons = _comparison_rows(seed_metrics)
    comparison_summary = _comparison_summary(comparisons)

    _write_csv(seed_metrics, output_dir / SEED_METRIC_FILE)
    _write_csv(stability, output_dir / STABILITY_FILE)
    _write_csv(comparisons, output_dir / COMPARISON_FILE)
    _write_csv(comparison_summary, output_dir / COMPARISON_SUMMARY_FILE)

    summary: dict[str, object] = {
        "status": "PASS",
        "analysis_scope": "fixed_setting_training_seed_stability",
        "analysis_role": "secondary_training_variation_analysis",
        "total_training_seed_count": TOTAL_SEED_COUNT,
        "existing_training_seed_count": EXISTING_SEED_COUNT,
        "additional_training_seed_count": ADDITIONAL_SEED_COUNT,
        "model_ids": list(MODEL_IDS),
        "seed_sets": {model: list(seeds) for model, seeds in SEEDS_BY_MODEL.items()},
        "model_selection_performed": False,
        "validation_search_repeated": False,
        "evaluation_labels_used_for_setting_choice": False,
        "existing_primary_predictions_changed": False,
        "additional_prediction_count": EXPECTED_ADDITIONAL_ROWS,
        "seed_metric_row_count": len(seed_metrics),
        "stability_row_count": len(stability),
        "comparison_row_count": len(comparisons),
        "comparison_summary_row_count": len(comparison_summary),
        "mean_direction_change_count": int(
            comparison_summary["mean_direction_changed"].sum()
        ),
        "outputs": [
            ADDITIONAL_PREDICTION_FILE,
            ADDITIONAL_HISTORY_FILE,
            SEED_METRIC_FILE,
            STABILITY_FILE,
            COMPARISON_FILE,
            COMPARISON_SUMMARY_FILE,
            SUMMARY_FILE,
            TRAINING_COMPLETE_FILE,
        ],
    }
    _write_json(summary, output_dir / SUMMARY_FILE)
    completion = {
        "status": "PASS",
        "additional_prediction_sha256": _sha256(
            output_dir / ADDITIONAL_PREDICTION_FILE
        ),
        "additional_history_sha256": _sha256(output_dir / ADDITIONAL_HISTORY_FILE),
        "summary_sha256": _sha256(output_dir / SUMMARY_FILE),
    }
    _write_json(completion, output_dir / TRAINING_COMPLETE_FILE)
    return summary


def run_training_seed_stability(
    repo_root: Path,
    *,
    device_name: str = "cuda",
    batch_size: int = stage32.BATCH_SIZE,
    postprocess_only: bool = False,
) -> Mapping[str, object]:
    repo_root = repo_root.resolve()
    if batch_size < 1:
        raise TrainingSeedStabilityError("Batch size must be positive")
    inspect_training_seed_stability_inputs(repo_root)
    stage32._configure_deterministic_execution()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise TrainingSeedStabilityError("CUDA is not available")
    device = torch.device(device_name)

    output_dir = repo_root / "data/interim" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = stage32._load_tables(repo_root)
    observations = stage32._build_analysis_observations(tables)
    contexts = stage32._make_contexts(tables, observations)
    selected_epochs = _selected_epoch_contract(repo_root)

    if postprocess_only:
        saved = _completed_additional_paths(output_dir)
        if saved is None:
            raise TrainingSeedStabilityError(
                "Postprocessing requires completed additional training-seed records"
            )
        additional_prediction_path, _ = saved
    else:
        additional_prediction_path, _ = _fit_additional_seeds(
            repo_root,
            output_dir,
            contexts,
            tables,
            selected_epochs,
            device,
            batch_size,
        )
    return _postprocess(repo_root, output_dir, additional_prediction_path)
