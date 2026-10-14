from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .sequence_neighbourhood_evaluation import (
    CONDITION_MODE_BY_SCHEME,
    CORE_SCHEME_IDS,
    SUBSTITUTION_INDEX,
    SUBSTITUTIONS,
    TARGET_COLUMN,
    _load_assay_sequences,
)
from .sequence_plm_embeddings import (
    _make_mutant_sequence,
    build_mutation_embedding_index,
)

MODEL_ID = "esm2_site_delta_ridge"

ESM2_SOURCE_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
ESM2_REQUESTED_REVISION = "main"
ESM2_HIDDEN_SIZE = 1280

EMBEDDING_HIDDEN_STATE_INDICES = (8, 16, 24, 33)
ESM2_RIDGE_ALPHAS = (
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
)

EXPECTED_PROTEINS = ("NDM-1", "VIM-2")

CHUNK_SIZE = 2048


class SequenceModelSensitivityError(RuntimeError):
    """Raised when ESM-2 mutation-effect evaluation cannot be completed safely."""


@dataclass
class ESM2Transform:
    hidden_state_index: int
    condition_mode: str
    condition_categories: tuple[str, ...]
    concentration_mean: float
    concentration_sd: float
    embedding_mean: np.ndarray
    embedding_sd: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def review_sequence_model_sensitivity_inputs(
    repo_root: Path,
    *,
    batch_size: int = 4,
) -> Mapping[str, object]:
    if batch_size < 1:
        raise SequenceModelSensitivityError("Batch size must be positive")
    embedding_index = build_mutation_embedding_index(repo_root)
    assay_sequences = _load_assay_sequences(repo_root)
    esmc_predictions = pd.read_csv(
        repo_root / "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv",
        dtype={"evaluation_iteration": str, "observation_id": str},
        low_memory=False,
    )
    esmc_predictions = esmc_predictions[
        esmc_predictions["model_id"].astype(str).eq("esmc_site_delta_ridge")
    ].copy()
    key = ["scheme_id", "evaluation_iteration", "evaluation_role", "observation_id"]
    if esmc_predictions.duplicated(key).any():
        raise SequenceModelSensitivityError(
            "Existing ESMC evaluation identities are not unique"
        )
    if len(esmc_predictions) != 135745:
        raise SequenceModelSensitivityError(
            f"Unexpected ESMC prediction count: {len(esmc_predictions)}"
        )
    if len(embedding_index) != 10086:
        raise SequenceModelSensitivityError(
            f"Unexpected unique mutation count: {len(embedding_index)}"
        )
    return {
        "status": "PASS",
        "analysis_role": "sequence_model_sensitivity_check",
        "primary_sequence_model_id": "esmc_site_delta_ridge",
        "sensitivity_model_id": MODEL_ID,
        "sensitivity_source_model_id": ESM2_SOURCE_MODEL_ID,
        "candidate_hidden_state_indices": list(EMBEDDING_HIDDEN_STATE_INDICES),
        "candidate_alphas": list(ESM2_RIDGE_ALPHAS),
        "mutation_count": int(len(embedding_index)),
        "existing_esmc_prediction_count": int(len(esmc_predictions)),
        "protein_ids": list(EXPECTED_PROTEINS),
        "assay_sequence_lengths": {
            protein_id: len(sequence)
            for protein_id, sequence in assay_sequences.items()
        },
        "batch_size": int(batch_size),
        "outcome_columns_loaded_for_embedding_inventory": False,
        "existing_esmc_predictions_changed": False,
        "model_selection_performed": False,
        "sequence_sensitivity_eval_output_exists": (
            repo_root / "data/interim/sequence_model_sensitivity/"
            "sequence_model_sensitivity_summary.json"
        ).is_file(),
    }


def prepare_esm2_embedding_index(
    repo_root: Path,
    batch_size: int = 4,
) -> Mapping[str, object]:
    if batch_size < 1:
        raise SequenceModelSensitivityError("Batch size must be positive")
    embedding_index = build_mutation_embedding_index(repo_root)
    assay_sequences = _load_assay_sequences(repo_root)
    output_dir = repo_root / "data/interim/sequence_model_sensitivity/esm2_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "mutation_embedding_index.csv"
    _write_csv(embedding_index, index_path)
    summary = {
        "model_id": ESM2_SOURCE_MODEL_ID,
        "requested_revision": ESM2_REQUESTED_REVISION,
        "model_revision": None,
        "precision": "float16",
        "batch_size": int(batch_size),
        "hidden_state_indices": list(EMBEDDING_HIDDEN_STATE_INDICES),
        "hidden_size": ESM2_HIDDEN_SIZE,
        "mutation_count": int(len(embedding_index)),
        "protein_counts": {
            str(protein_id): int(count)
            for protein_id, count in (
                embedding_index.groupby("protein_id").size().items()
            )
        },
        "assay_sequence_lengths": {
            protein_id: len(sequence)
            for protein_id, sequence in assay_sequences.items()
        },
        "assay_sequence_sha256": {
            protein_id: _sha256_text(sequence)
            for protein_id, sequence in assay_sequences.items()
        },
        "embedding_index_sha256": _sha256_file(index_path),
        "outcome_columns_loaded": False,
        "site_embedding_definition": (
            "ESM-2 hidden states at the mutated assay-residue token for each mutant; "
            "wild-type residue states are extracted once per protein"
        ),
        "candidate_hidden_state_policy": (
            "approximately quarter-depth states 8, 16, 24, and final state 33; "
            "the candidate set is fixed before evaluation"
        ),
    }
    (output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _resolve_esm2_snapshot(local_files_only: bool) -> tuple[Path, str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SequenceModelSensitivityError(
            "huggingface-hub is required for ESM-2 extraction"
        ) from exc
    snapshot = Path(
        snapshot_download(
            repo_id=ESM2_SOURCE_MODEL_ID,
            revision=ESM2_REQUESTED_REVISION,
            local_files_only=local_files_only,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
            ],
        )
    )
    revision = snapshot.name
    if len(revision) < 7:
        raise SequenceModelSensitivityError(
            "Unable to record the ESM-2 snapshot revision"
        )
    return snapshot, revision


def _progress_record(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_esm2_embeddings(
    repo_root: Path,
    *,
    batch_size: int = 4,
    local_files_only: bool = True,
) -> Mapping[str, object]:
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SequenceModelSensitivityError(
            "ESM-2 extraction requires torch and transformers"
        ) from exc
    if not torch.cuda.is_available():
        raise SequenceModelSensitivityError("CUDA is required for ESM-2 extraction")
    summary = prepare_esm2_embedding_index(repo_root, batch_size=batch_size)
    embedding_index = build_mutation_embedding_index(repo_root)
    assay_sequences = _load_assay_sequences(repo_root)
    output_dir = repo_root / "data/interim/sequence_model_sensitivity/esm2_embeddings"
    mutant_path = output_dir / "mutant_site_embeddings.npy"
    wt_path = output_dir / "wild_type_residue_embeddings.npy"
    extraction_path = output_dir / "extraction_summary.json"
    progress_path = output_dir / "extraction_progress.json"
    mutant_tmp = output_dir / "mutant_site_embeddings.partial.npy"
    wt_tmp = output_dir / "wild_type_residue_embeddings.partial.npy"
    if mutant_path.is_file() and wt_path.is_file() and extraction_path.is_file():
        return json.loads(extraction_path.read_text(encoding="utf-8"))
    snapshot, revision = _resolve_esm2_snapshot(local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    if int(model.config.hidden_size) != ESM2_HIDDEN_SIZE:
        raise SequenceModelSensitivityError("Unexpected ESM-2 hidden size")
    if int(model.config.num_hidden_layers) != 33:
        raise SequenceModelSensitivityError("Unexpected ESM-2 layer count")
    protein_order = EXPECTED_PROTEINS
    max_length = max(len(assay_sequences[p]) for p in protein_order)
    progress = _progress_record(progress_path)
    completed_rows = 0
    wt_complete = False
    if progress is not None:
        if progress.get("model_revision") != revision:
            raise SequenceModelSensitivityError(
                "ESM-2 extraction revision changed during resume"
            )
        if int(progress.get("batch_size", -1)) != int(batch_size):
            raise SequenceModelSensitivityError(
                "ESM-2 extraction batch size changed during resume"
            )
        completed_rows = int(progress.get("completed_mutation_rows", 0))
        wt_complete = bool(progress.get("wild_type_complete", False))
    if completed_rows < 0 or completed_rows > len(embedding_index):
        raise SequenceModelSensitivityError("Invalid ESM-2 extraction progress")
    if completed_rows and not mutant_tmp.is_file():
        raise SequenceModelSensitivityError("ESM-2 mutant partial array is missing")
    if wt_complete and not wt_tmp.is_file():
        raise SequenceModelSensitivityError("ESM-2 wild-type partial array is missing")
    if not wt_complete:
        wt_array = np.lib.format.open_memmap(
            wt_tmp,
            mode="w+",
            dtype=np.float16,
            shape=(
                len(protein_order),
                len(EMBEDDING_HIDDEN_STATE_INDICES),
                max_length,
                ESM2_HIDDEN_SIZE,
            ),
        )
        wt_array[:] = np.nan
        with torch.inference_mode():
            for protein_index, protein_id in enumerate(protein_order):
                sequence = assay_sequences[protein_id]
                encoded = tokenizer(
                    sequence,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                tokens = tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"][0].tolist()
                )
                if tokens[1 : len(sequence) + 1] != list(sequence):
                    raise SequenceModelSensitivityError(
                        f"ESM-2 tokenizer residue contract failed for {protein_id}"
                    )
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                output = model(**encoded, output_hidden_states=True, return_dict=True)
                states = output.hidden_states
                if states is None or len(states) != 34:
                    raise SequenceModelSensitivityError(
                        "Unexpected ESM-2 hidden-state output"
                    )
                for layer_index, state_index in enumerate(
                    EMBEDDING_HIDDEN_STATE_INDICES
                ):
                    values = states[state_index][0, 1 : len(sequence) + 1, :]
                    if not bool(torch.isfinite(values).all()):
                        raise SequenceModelSensitivityError(
                            "Non-finite ESM-2 wild-type state"
                        )
                    wt_array[protein_index, layer_index, : len(sequence), :] = (
                        values.detach().cpu().numpy().astype(np.float16, copy=False)
                    )
        wt_array.flush()
        del wt_array
        wt_complete = True
        progress_path.write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "model_revision": revision,
                    "batch_size": int(batch_size),
                    "wild_type_complete": True,
                    "completed_mutation_rows": int(completed_rows),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if completed_rows == 0:
        mutant_array = np.lib.format.open_memmap(
            mutant_tmp,
            mode="w+",
            dtype=np.float16,
            shape=(
                len(embedding_index),
                len(EMBEDDING_HIDDEN_STATE_INDICES),
                ESM2_HIDDEN_SIZE,
            ),
        )
    else:
        mutant_array = np.lib.format.open_memmap(mutant_tmp, mode="r+")
    peak_allocated = 0
    with torch.inference_mode():
        for start in range(completed_rows, len(embedding_index), batch_size):
            stop = min(start + batch_size, len(embedding_index))
            batch = embedding_index.iloc[start:stop]
            mutant_sequences = []
            positions = []
            for row in batch.itertuples(index=False):
                sequence = assay_sequences[str(row.protein_id)]
                mutant_sequences.append(
                    _make_mutant_sequence(
                        sequence,
                        int(row.assay_position),
                        str(row.wild_type),
                        str(row.mutant),
                    )
                )
                positions.append(int(row.assay_position))
            encoded = tokenizer(
                mutant_sequences,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            token_rows = [
                tokenizer.convert_ids_to_tokens(encoded["input_ids"][index].tolist())
                for index in range(len(batch))
            ]
            for local_index, row in enumerate(batch.itertuples(index=False)):
                position = int(row.assay_position)
                if token_rows[local_index][position] != str(row.mutant):
                    raise SequenceModelSensitivityError(
                        "ESM-2 mutant token mismatch for "
                        f"{row.protein_id}/{row.assay_mutation}"
                    )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, return_dict=True)
            states = output.hidden_states
            if states is None or len(states) != 34:
                raise SequenceModelSensitivityError(
                    "Unexpected ESM-2 mutant hidden states"
                )
            for local_index, position in enumerate(positions):
                row_index = start + local_index
                for layer_index, state_index in enumerate(
                    EMBEDDING_HIDDEN_STATE_INDICES
                ):
                    vector = states[state_index][local_index, position, :]
                    if not bool(torch.isfinite(vector).all()):
                        raise SequenceModelSensitivityError(
                            "Non-finite ESM-2 mutant state"
                        )
                    mutant_array[row_index, layer_index, :] = (
                        vector.detach().cpu().numpy().astype(np.float16, copy=False)
                    )
            mutant_array.flush()
            completed_rows = stop
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
            progress_path.write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "model_revision": revision,
                        "batch_size": int(batch_size),
                        "wild_type_complete": True,
                        "completed_mutation_rows": int(completed_rows),
                        "mutation_count": int(len(embedding_index)),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"ESM-2 mutation embeddings: {completed_rows}/{len(embedding_index)}",
                flush=True,
            )
    del mutant_array
    os.replace(mutant_tmp, mutant_path)
    os.replace(wt_tmp, wt_path)
    summary = dict(summary)
    summary["model_revision"] = revision
    (output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    extraction = {
        "status": "PASS",
        "model_id": ESM2_SOURCE_MODEL_ID,
        "model_revision": revision,
        "requested_revision": ESM2_REQUESTED_REVISION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "precision": "float16",
        "batch_size": int(batch_size),
        "hidden_state_indices": list(EMBEDDING_HIDDEN_STATE_INDICES),
        "hidden_size": ESM2_HIDDEN_SIZE,
        "mutation_count": int(len(embedding_index)),
        "embedding_index_sha256": _sha256_file(
            output_dir / "mutation_embedding_index.csv"
        ),
        "mutant_embedding_shape": [
            int(len(embedding_index)),
            len(EMBEDDING_HIDDEN_STATE_INDICES),
            ESM2_HIDDEN_SIZE,
        ],
        "wild_type_embedding_shape": [
            len(protein_order),
            len(EMBEDDING_HIDDEN_STATE_INDICES),
            max_length,
            ESM2_HIDDEN_SIZE,
        ],
        "peak_allocated_bytes": int(peak_allocated),
        "outcome_columns_loaded": False,
        "token_index_contract": "one_based_assay_position_equals_token_index",
        "resume_policy": (
            "completed mutation batches are reused; an interrupted batch is rerun"
        ),
    }
    extraction_path.write_text(
        json.dumps(extraction, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress_path.exists():
        progress_path.unlink()
    return extraction


def _validate_embedding_record(
    repo_root: Path,
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> None:
    embedding_dir = (
        repo_root / "data/interim/sequence_model_sensitivity/esm2_embeddings"
    )
    index_path = embedding_dir / "mutation_embedding_index.csv"

    embedding_summary = json.loads(
        (embedding_dir / "embedding_summary.json").read_text(encoding="utf-8")
    )
    extraction_summary = json.loads(
        (embedding_dir / "extraction_summary.json").read_text(encoding="utf-8")
    )

    for name, summary in (
        ("embedding summary", embedding_summary),
        ("extraction summary", extraction_summary),
    ):
        if summary.get("model_id") != ESM2_SOURCE_MODEL_ID:
            raise SequenceModelSensitivityError(
                f"{name} has an unexpected ESM-2 model identifier"
            )
        if list(summary.get("hidden_state_indices", [])) != list(
            EMBEDDING_HIDDEN_STATE_INDICES
        ):
            raise SequenceModelSensitivityError(
                f"{name} has unexpected ESM-2 hidden-state indices"
            )
        if int(summary.get("hidden_size", -1)) != ESM2_HIDDEN_SIZE:
            raise SequenceModelSensitivityError(
                f"{name} has an unexpected ESM-2 hidden size"
            )
        if int(summary.get("mutation_count", -1)) != len(embedding_index):
            raise SequenceModelSensitivityError(
                f"{name} mutation count does not match the embedding index"
            )
        if summary.get("precision") != "float16":
            raise SequenceModelSensitivityError(
                f"{name} has an unexpected embedding precision"
            )
        if summary.get("outcome_columns_loaded") is not False:
            raise SequenceModelSensitivityError(
                f"{name} does not establish outcome-blind extraction"
            )

    summary_revision = str(embedding_summary.get("model_revision") or "")
    extraction_revision = str(extraction_summary.get("model_revision") or "")
    if not summary_revision or summary_revision != extraction_revision:
        raise SequenceModelSensitivityError("ESM-2 model revision records disagree")

    actual_index_sha256 = _sha256_file(index_path)
    summary_index_sha256 = str(embedding_summary.get("embedding_index_sha256", ""))
    extraction_index_sha256 = str(extraction_summary.get("embedding_index_sha256", ""))

    if not summary_index_sha256 or not extraction_index_sha256:
        raise SequenceModelSensitivityError(
            "Embedding-index SHA-256 is missing from extraction record"
        )
    if summary_index_sha256 != extraction_index_sha256:
        raise SequenceModelSensitivityError(
            "Embedding-index SHA-256 disagrees between extraction records"
        )
    if actual_index_sha256 != summary_index_sha256:
        raise SequenceModelSensitivityError(
            "Embedding-index SHA-256 does not match extraction record"
        )

    assay_sequences = _load_assay_sequences(repo_root)

    if set(assay_sequences) != set(EXPECTED_PROTEINS):
        raise SequenceModelSensitivityError(
            "Unexpected proteins in the assay-sequence contract"
        )

    recorded_lengths = embedding_summary.get("assay_sequence_lengths")
    recorded_hashes = embedding_summary.get("assay_sequence_sha256")

    if not isinstance(recorded_lengths, dict) or not isinstance(recorded_hashes, dict):
        raise SequenceModelSensitivityError(
            "Embedding summary lacks assay-sequence record"
        )

    for protein_id in EXPECTED_PROTEINS:
        sequence = assay_sequences[protein_id]
        if int(recorded_lengths.get(protein_id, -1)) != len(sequence):
            raise SequenceModelSensitivityError(
                f"Assay-sequence length disagrees for {protein_id}"
            )
        if recorded_hashes.get(protein_id) != _sha256_text(sequence):
            raise SequenceModelSensitivityError(
                f"Assay-sequence SHA-256 disagrees for {protein_id}"
            )

    expected_mutant_shape = (
        len(embedding_index),
        len(EMBEDDING_HIDDEN_STATE_INDICES),
        ESM2_HIDDEN_SIZE,
    )
    if mutant_embeddings.shape != expected_mutant_shape:
        raise SequenceModelSensitivityError(
            f"Unexpected mutant embedding shape: {mutant_embeddings.shape}"
        )

    expected_wild_type_shape = (
        len(EXPECTED_PROTEINS),
        len(EMBEDDING_HIDDEN_STATE_INDICES),
        max(len(assay_sequences[protein]) for protein in EXPECTED_PROTEINS),
        ESM2_HIDDEN_SIZE,
    )
    if wild_type_embeddings.shape != expected_wild_type_shape:
        raise SequenceModelSensitivityError(
            f"Unexpected wild-type embedding shape: {wild_type_embeddings.shape}"
        )

    if not np.isfinite(mutant_embeddings).all():
        raise SequenceModelSensitivityError(
            "Mutant embedding array contains non-finite values"
        )

    for protein_index, protein_id in enumerate(EXPECTED_PROTEINS):
        length = len(assay_sequences[protein_id])
        used = wild_type_embeddings[protein_index, :, :length, :]
        if not np.isfinite(used).all():
            raise SequenceModelSensitivityError(
                f"Wild-type embedding array contains non-finite values for {protein_id}"
            )


def _condition_token(frame: pd.DataFrame) -> pd.Series:
    concentration = frame["concentration"].map(
        lambda value: "" if pd.isna(value) else format(float(value), ".12g")
    )
    temperature = frame["temperature_c"].map(
        lambda value: "" if pd.isna(value) else format(float(value), ".12g")
    )
    return (
        frame["antibiotic"].fillna("").astype(str)
        + "|"
        + concentration
        + "|"
        + temperature
    )


def _load_inputs(
    repo_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    observations = pd.read_csv(
        repo_root / "data/processed/missense_observations.csv",
        low_memory=False,
    )
    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv",
        low_memory=False,
    )
    assignments = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/group_assignments.csv",
        low_memory=False,
    )
    membership = pd.read_csv(
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv",
        low_memory=False,
    )

    embedding_dir = (
        repo_root / "data/interim/sequence_model_sensitivity/esm2_embeddings"
    )

    embedding_index = pd.read_csv(
        embedding_dir / "mutation_embedding_index.csv",
        low_memory=False,
    )
    mutant_embeddings = np.load(
        embedding_dir / "mutant_site_embeddings.npy",
        mmap_mode="r",
    )
    wild_type_embeddings = np.load(
        embedding_dir / "wild_type_residue_embeddings.npy",
        mmap_mode="r",
    )

    return (
        observations,
        schemes,
        assignments,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )


def _validate_inputs(
    observations: pd.DataFrame,
    membership: pd.DataFrame,
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> None:
    if TARGET_COLUMN not in observations:
        raise SequenceModelSensitivityError(f"Missing outcome column: {TARGET_COLUMN}")
    if observations[TARGET_COLUMN].isna().any():
        raise SequenceModelSensitivityError("Outcome column contains missing values")

    if TARGET_COLUMN in membership.columns or "score_original" in membership.columns:
        raise SequenceModelSensitivityError("Outcome leaked into modelling membership")

    required_index_columns = {
        "embedding_row",
        "protein_id",
        "assay_mutation",
        "assay_position",
        "wild_type",
        "mutant",
        "mutant_sequence_sha256",
    }
    missing = required_index_columns.difference(embedding_index.columns)
    if missing:
        raise SequenceModelSensitivityError(
            f"Embedding index is missing columns: {sorted(missing)}"
        )

    if embedding_index.duplicated(["protein_id", "assay_mutation"]).any():
        raise SequenceModelSensitivityError(
            "Embedding index contains duplicate protein/mutation rows"
        )

    expected_rows = np.arange(len(embedding_index), dtype=int)
    try:
        actual_rows = embedding_index["embedding_row"].to_numpy(dtype=int)
    except (TypeError, ValueError) as exc:
        raise SequenceModelSensitivityError(
            "Embedding rows must be integer-valued"
        ) from exc

    if not np.array_equal(actual_rows, expected_rows):
        raise SequenceModelSensitivityError(
            "Embedding rows must form the contiguous range 0..N-1"
        )

    proteins = set(embedding_index["protein_id"].astype(str))
    if proteins != set(EXPECTED_PROTEINS):
        raise SequenceModelSensitivityError(
            f"Unexpected embedding-index proteins: {sorted(proteins)}"
        )


def _join_scheme_observations(
    membership: pd.DataFrame,
    observations: pd.DataFrame,
    embedding_index: pd.DataFrame,
    scheme_id: str,
) -> pd.DataFrame:
    scheme_membership = membership[membership["scheme_id"].eq(scheme_id)].copy()

    if scheme_membership.empty:
        raise SequenceModelSensitivityError(f"No modelling rows for {scheme_id}")

    observation_columns = [
        "observation_id",
        "protein_id",
        "assay_mutation",
        "assay_position",
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
        TARGET_COLUMN,
    ]

    frame = scheme_membership.merge(
        observations[observation_columns],
        on=["observation_id", "protein_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_observation"),
    )

    if frame[TARGET_COLUMN].isna().any():
        raise SequenceModelSensitivityError(
            f"Missing outcomes after joining {scheme_id}"
        )

    frame = frame.merge(
        embedding_index[
            [
                "embedding_row",
                "protein_id",
                "assay_mutation",
                "assay_position",
                "wild_type",
                "mutant",
            ]
        ],
        on=[
            "protein_id",
            "assay_mutation",
            "assay_position",
            "wild_type",
            "mutant",
        ],
        how="left",
        validate="many_to_one",
    )

    if frame["embedding_row"].isna().any():
        raise SequenceModelSensitivityError(f"Missing ESM-2 embeddings for {scheme_id}")

    frame["embedding_row"] = frame["embedding_row"].astype(int)
    return frame


def _hidden_state_array_index(hidden_state_index: int) -> int:
    try:
        return EMBEDDING_HIDDEN_STATE_INDICES.index(int(hidden_state_index))
    except ValueError as exc:
        raise SequenceModelSensitivityError(
            f"Unsupported ESM-2 hidden-state index: {hidden_state_index}"
        ) from exc


def _site_delta(
    frame: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    *,
    hidden_state_index: int,
) -> np.ndarray:
    layer_array_index = _hidden_state_array_index(hidden_state_index)
    embedding_rows = frame["embedding_row"].to_numpy(dtype=int)
    positions = frame["assay_position"].to_numpy(dtype=int) - 1
    proteins = frame["protein_id"].astype(str).to_numpy()

    unexpected_proteins = sorted(set(proteins).difference(EXPECTED_PROTEINS))
    if unexpected_proteins:
        raise SequenceModelSensitivityError(
            f"Unexpected protein identifiers in ESM-2 rows: {unexpected_proteins}"
        )

    protein_index = {
        protein_id: index for index, protein_id in enumerate(EXPECTED_PROTEINS)
    }
    protein_indices = np.asarray(
        [protein_index[protein_id] for protein_id in proteins],
        dtype=int,
    )

    sequence_lengths = {
        "NDM-1": wild_type_embeddings.shape[2],
        "VIM-2": 267,
    }
    for protein_id in EXPECTED_PROTEINS:
        mask = proteins == protein_id
        if mask.any():
            invalid = (positions[mask] < 0) | (
                positions[mask] >= sequence_lengths[protein_id]
            )
            if invalid.any():
                raise SequenceModelSensitivityError(
                    f"ESM-2 assay position outside {protein_id} sequence"
                )

    mutant = np.asarray(
        mutant_embeddings[
            embedding_rows,
            layer_array_index,
            :,
        ],
        dtype=np.float32,
    )

    wild_type = np.asarray(
        wild_type_embeddings[
            protein_indices,
            layer_array_index,
            positions,
            :,
        ],
        dtype=np.float32,
    )

    delta = mutant - wild_type

    if not np.isfinite(delta).all():
        raise SequenceModelSensitivityError("Non-finite ESM-2 site-delta values")

    return delta


def _fit_transform(
    train: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    *,
    condition_mode: str,
    hidden_state_index: int,
) -> ESM2Transform:
    delta = _site_delta(
        train,
        mutant_embeddings,
        wild_type_embeddings,
        hidden_state_index=hidden_state_index,
    ).astype(np.float64)

    mean = delta.mean(axis=0)
    sd = delta.std(axis=0, ddof=0)

    bad_sd = ~np.isfinite(sd) | (sd == 0.0)
    sd[bad_sd] = 1.0

    categories: tuple[str, ...] = ()
    concentration_mean = 0.0
    concentration_sd = 1.0

    if condition_mode == "categorical_condition":
        categories = tuple(sorted(_condition_token(train).unique()))

    elif condition_mode == "log2_concentration":
        concentration = train["concentration"].to_numpy(dtype=float)
        if not np.isfinite(concentration).all() or (concentration <= 0).any():
            raise SequenceModelSensitivityError(
                "Positive finite concentration is required"
            )

        values = np.log2(concentration)
        concentration_mean = float(values.mean())
        concentration_sd = float(values.std(ddof=0))

        if not math.isfinite(concentration_sd) or concentration_sd == 0.0:
            concentration_sd = 1.0

    elif condition_mode != "none":
        raise SequenceModelSensitivityError(f"Unknown condition mode: {condition_mode}")

    return ESM2Transform(
        hidden_state_index=hidden_state_index,
        condition_mode=condition_mode,
        condition_categories=categories,
        concentration_mean=concentration_mean,
        concentration_sd=concentration_sd,
        embedding_mean=mean,
        embedding_sd=sd,
    )


def _feature_count(transform: ESM2Transform) -> int:
    count = 1 + len(SUBSTITUTIONS)

    if transform.condition_mode == "categorical_condition":
        count += len(transform.condition_categories)
    elif transform.condition_mode == "log2_concentration":
        count += 1

    count += ESM2_HIDDEN_SIZE
    return count


def _design_chunk(
    frame: pd.DataFrame,
    transform: ESM2Transform,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> np.ndarray:
    feature_count = _feature_count(transform)
    matrix = np.zeros(
        (len(frame), feature_count),
        dtype=np.float64,
    )

    matrix[:, 0] = 1.0
    substitution_offset = 1

    for row_number, row in enumerate(frame.itertuples(index=False)):
        substitution = (str(row.wild_type), str(row.mutant))
        if substitution not in SUBSTITUTION_INDEX:
            raise SequenceModelSensitivityError(
                f"Invalid missense substitution: {substitution}"
            )

        matrix[
            row_number,
            substitution_offset + SUBSTITUTION_INDEX[substitution],
        ] = 1.0

    condition_offset = substitution_offset + len(SUBSTITUTIONS)
    embedding_offset = condition_offset

    if transform.condition_mode == "categorical_condition":
        condition_index = {
            value: index for index, value in enumerate(transform.condition_categories)
        }
        tokens = _condition_token(frame).tolist()

        for row_number, token in enumerate(tokens):
            if token in condition_index:
                matrix[
                    row_number,
                    condition_offset + condition_index[token],
                ] = 1.0

        embedding_offset += len(transform.condition_categories)

    elif transform.condition_mode == "log2_concentration":
        concentration = frame["concentration"].to_numpy(dtype=float)
        if not np.isfinite(concentration).all() or (concentration <= 0).any():
            raise SequenceModelSensitivityError(
                "Positive finite concentration is required"
            )

        matrix[:, condition_offset] = (
            np.log2(concentration) - transform.concentration_mean
        ) / transform.concentration_sd

        embedding_offset += 1

    delta = _site_delta(
        frame,
        mutant_embeddings,
        wild_type_embeddings,
        hidden_state_index=transform.hidden_state_index,
    ).astype(np.float64)

    matrix[:, embedding_offset:] = (
        delta - transform.embedding_mean
    ) / transform.embedding_sd

    if not np.isfinite(matrix).all():
        raise SequenceModelSensitivityError("Non-finite ESM-2 design matrix")

    return matrix


def _ridge_statistics(
    train: pd.DataFrame,
    transform: ESM2Transform,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = _feature_count(transform)
    gram = np.zeros((feature_count, feature_count), dtype=np.float64)
    target_product = np.zeros(feature_count, dtype=np.float64)

    for start in range(0, len(train), CHUNK_SIZE):
        chunk = train.iloc[start : start + CHUNK_SIZE]
        matrix = _design_chunk(
            chunk,
            transform,
            mutant_embeddings,
            wild_type_embeddings,
        )
        target = chunk[TARGET_COLUMN].to_numpy(dtype=np.float64)

        with np.errstate(
            divide="ignore",
            over="ignore",
            invalid="ignore",
        ):
            gram_update = matrix.T @ matrix
            target_update = matrix.T @ target

        if not np.isfinite(gram_update).all():
            raise SequenceModelSensitivityError("Non-finite ridge Gram-matrix update")
        if not np.isfinite(target_update).all():
            raise SequenceModelSensitivityError(
                "Non-finite ridge target-product update"
            )

        gram += gram_update
        target_product += target_update

    if not np.isfinite(gram).all():
        raise SequenceModelSensitivityError("Non-finite ridge Gram matrix")
    if not np.isfinite(target_product).all():
        raise SequenceModelSensitivityError("Non-finite ridge target product")

    return gram, target_product


def _ridge_coefficients(
    gram: np.ndarray,
    target_product: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise SequenceModelSensitivityError("Ridge alpha must be positive and finite")

    penalty = np.eye(len(target_product), dtype=np.float64)
    penalty[0, 0] = 0.0

    coefficients = np.linalg.solve(
        gram + float(alpha) * penalty,
        target_product,
    )

    if not np.isfinite(coefficients).all():
        raise SequenceModelSensitivityError("Ridge produced non-finite coefficients")

    return coefficients


def _predict(
    frame: pd.DataFrame,
    transform: ESM2Transform,
    coefficients: np.ndarray,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> np.ndarray:
    predictions = np.empty(len(frame), dtype=np.float64)

    for start in range(0, len(frame), CHUNK_SIZE):
        chunk = frame.iloc[start : start + CHUNK_SIZE]
        matrix = _design_chunk(
            chunk,
            transform,
            mutant_embeddings,
            wild_type_embeddings,
        )
        with np.errstate(
            divide="ignore",
            over="ignore",
            invalid="ignore",
        ):
            prediction_chunk = matrix @ coefficients

        if not np.isfinite(prediction_chunk).all():
            raise SequenceModelSensitivityError("Non-finite ESM-2 prediction")

        predictions[start : start + len(chunk)] = prediction_chunk

    if not np.isfinite(predictions).all():
        raise SequenceModelSensitivityError("Non-finite ESM-2 predictions")

    return predictions


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    *,
    condition_mode: str,
    hidden_state_index: int,
    alpha: float,
) -> np.ndarray:
    transform = _fit_transform(
        train,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
        hidden_state_index=hidden_state_index,
    )

    gram, target_product = _ridge_statistics(
        train,
        transform,
        mutant_embeddings,
        wild_type_embeddings,
    )

    coefficients = _ridge_coefficients(
        gram,
        target_product,
        alpha=alpha,
    )

    return _predict(
        test,
        transform,
        coefficients,
        mutant_embeddings,
        wild_type_embeddings,
    )


def _select_model_setting(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    *,
    condition_mode: str,
) -> tuple[int, float, float, list[Mapping[str, object]]]:
    detail: list[Mapping[str, object]] = []
    candidates: list[tuple[float, int, float]] = []

    for hidden_state_index in EMBEDDING_HIDDEN_STATE_INDICES:
        fold_data = []

        for train, validation in folds:
            transform = _fit_transform(
                train,
                mutant_embeddings,
                wild_type_embeddings,
                condition_mode=condition_mode,
                hidden_state_index=hidden_state_index,
            )

            gram, target_product = _ridge_statistics(
                train,
                transform,
                mutant_embeddings,
                wild_type_embeddings,
            )

            fold_data.append(
                (
                    validation,
                    transform,
                    gram,
                    target_product,
                )
            )

        for alpha in ESM2_RIDGE_ALPHAS:
            fold_maes = []

            for fold_index, (
                validation,
                transform,
                gram,
                target_product,
            ) in enumerate(fold_data):
                coefficients = _ridge_coefficients(
                    gram,
                    target_product,
                    alpha=alpha,
                )

                predictions = _predict(
                    validation,
                    transform,
                    coefficients,
                    mutant_embeddings,
                    wild_type_embeddings,
                )

                mae = float(
                    np.mean(
                        np.abs(
                            validation[TARGET_COLUMN].to_numpy(dtype=float)
                            - predictions
                        )
                    )
                )

                fold_maes.append(mae)

                detail.append(
                    {
                        "model_id": MODEL_ID,
                        "hidden_state_index": hidden_state_index,
                        "alpha": alpha,
                        "cv_fold": fold_index,
                        "validation_mae": mae,
                    }
                )

            candidates.append(
                (
                    float(np.mean(fold_maes)),
                    hidden_state_index,
                    alpha,
                )
            )

    best_mae, best_hidden_state_index, best_alpha = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )

    return best_hidden_state_index, best_alpha, best_mae, detail


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Mapping[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise SequenceModelSensitivityError("Metrics require equal non-empty arrays")

    error = y_pred - y_true
    true_constant = np.allclose(
        y_true,
        y_true[0],
        rtol=0.0,
        atol=1e-12,
    )
    pred_constant = np.allclose(
        y_pred,
        y_pred[0],
        rtol=0.0,
        atol=1e-12,
    )

    if true_constant or pred_constant:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

        true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
        pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)

        spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])

    return {
        "spearman": spearman,
        "pearson": pearson,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _prediction_records(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    scheme: pd.Series,
    evaluation_iteration: object,
    evaluation_role: str,
    hidden_state_index: int,
    alpha: float,
) -> list[Mapping[str, object]]:
    result = pd.DataFrame(
        {
            "scheme_id": str(scheme["scheme_id"]),
            "task_id": str(scheme["task_id"]),
            "model_id": MODEL_ID,
            "hidden_state_index": hidden_state_index,
            "evaluation_iteration": evaluation_iteration,
            "evaluation_role": evaluation_role,
            "observation_id": test["observation_id"].to_numpy(),
            "protein_id": test["protein_id"].to_numpy(),
            "assay_position": test["assay_position"].to_numpy(dtype=int),
            "wild_type": test["wild_type"].to_numpy(),
            "mutant": test["mutant"].to_numpy(),
            "antibiotic": test["antibiotic"].to_numpy(),
            "concentration": test["concentration"].to_numpy(),
            "temperature_c": test["temperature_c"].to_numpy(),
            "evaluation_group_id": test["group_id"].to_numpy(),
            "y_true": test[TARGET_COLUMN].to_numpy(dtype=float),
            "y_pred": predictions,
            "selected_alpha": alpha,
        }
    )

    return result.to_dict(orient="records")


def _nested_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    scheme_id = str(scheme["scheme_id"])
    condition_mode = CONDITION_MODE_BY_SCHEME[scheme_id]

    scheme_assignments = assignments[assignments["scheme_id"].eq(scheme_id)]

    predictions: list[Mapping[str, object]] = []
    selections: list[Mapping[str, object]] = []

    for iteration in sorted(
        scheme_assignments["iteration"].dropna().astype(int).unique()
    ):
        current = scheme_assignments[scheme_assignments["iteration"].eq(iteration)]

        role_map = dict(
            zip(
                current["group_id"],
                current["split_role"],
            )
        )
        roles = frame["group_id"].map(role_map)

        train = frame[roles.eq("train")].copy()
        validation = frame[roles.eq("validation")].copy()
        test = frame[roles.eq("test")].copy()

        if train.empty or validation.empty or test.empty:
            raise SequenceModelSensitivityError(
                f"Incomplete nested split for {scheme_id}/{iteration}"
            )

        hidden_state_index, alpha, validation_mae, detail = _select_model_setting(
            [(train, validation)],
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=condition_mode,
        )

        for row in detail:
            selections.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": str(scheme["task_id"]),
                    "evaluation_iteration": iteration,
                    "selection_scope": "inner_validation",
                    **row,
                    "selected": (
                        row["hidden_state_index"] == hidden_state_index
                        and row["alpha"] == alpha
                    ),
                }
            )

        selections.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme["task_id"]),
                "evaluation_iteration": iteration,
                "selection_scope": "selected_setting",
                "model_id": MODEL_ID,
                "hidden_state_index": hidden_state_index,
                "alpha": alpha,
                "cv_fold": "",
                "validation_mae": validation_mae,
                "selected": True,
            }
        )

        refit_data = pd.concat(
            [train, validation],
            ignore_index=True,
        )

        y_pred = _fit_predict(
            refit_data,
            test,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=condition_mode,
            hidden_state_index=hidden_state_index,
            alpha=alpha,
        )

        predictions.extend(
            _prediction_records(
                test,
                y_pred,
                scheme=scheme,
                evaluation_iteration=iteration,
                evaluation_role="outer_test",
                hidden_state_index=hidden_state_index,
                alpha=alpha,
            )
        )

    return predictions, selections


def _source_cv_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    scheme_id = str(scheme["scheme_id"])
    condition_mode = CONDITION_MODE_BY_SCHEME[scheme_id]

    scheme_assignments = assignments[assignments["scheme_id"].eq(scheme_id)]

    source_assignments = scheme_assignments[
        scheme_assignments["split_role"].eq("source_model_selection")
    ]
    target_assignments = scheme_assignments[
        scheme_assignments["split_role"].eq("target_test")
    ]

    source = frame[frame["scope_role"].eq("source_model_selection")].copy()
    target = frame[frame["scope_role"].eq("target_test")].copy()

    source["source_cv_fold"] = source["group_id"].map(
        dict(
            zip(
                source_assignments["group_id"],
                source_assignments["cv_fold"],
            )
        )
    )

    if source["source_cv_fold"].isna().any():
        raise SequenceModelSensitivityError(
            f"Unassigned source CV rows for {scheme_id}"
        )

    if set(target["group_id"]) != set(target_assignments["group_id"]):
        raise SequenceModelSensitivityError(
            f"Target-test group mismatch for {scheme_id}"
        )

    folds = []

    for fold in sorted(source["source_cv_fold"].astype(int).unique()):
        folds.append(
            (
                source[~source["source_cv_fold"].eq(fold)].copy(),
                source[source["source_cv_fold"].eq(fold)].copy(),
            )
        )

    hidden_state_index, alpha, validation_mae, detail = _select_model_setting(
        folds,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
    )

    selections: list[Mapping[str, object]] = []

    for row in detail:
        selections.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme["task_id"]),
                "evaluation_iteration": "target",
                "selection_scope": "source_cv",
                **row,
                "selected": (
                    row["hidden_state_index"] == hidden_state_index
                    and row["alpha"] == alpha
                ),
            }
        )

    selections.append(
        {
            "scheme_id": scheme_id,
            "task_id": str(scheme["task_id"]),
            "evaluation_iteration": "target",
            "selection_scope": "selected_setting",
            "model_id": MODEL_ID,
            "hidden_state_index": hidden_state_index,
            "alpha": alpha,
            "cv_fold": "",
            "validation_mae": validation_mae,
            "selected": True,
        }
    )

    y_pred = _fit_predict(
        source,
        target,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
        hidden_state_index=hidden_state_index,
        alpha=alpha,
    )

    predictions = _prediction_records(
        target,
        y_pred,
        scheme=scheme,
        evaluation_iteration="target",
        evaluation_role="target_test",
        hidden_state_index=hidden_state_index,
        alpha=alpha,
    )

    return predictions, selections


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (scheme_id, model_id), group in predictions.groupby(
        ["scheme_id", "model_id"],
        sort=True,
    ):
        subsets = list(
            group.groupby(
                "evaluation_iteration",
                sort=True,
            )
        )
        subsets.append(("pooled", group))

        for evaluation_iteration, subset in subsets:
            hidden_states = sorted(subset["hidden_state_index"].unique().tolist())
            metric_hidden_state: int | str
            if len(hidden_states) == 1:
                metric_hidden_state = int(hidden_states[0])
            else:
                metric_hidden_state = "mixed"

            rows.append(
                {
                    "scheme_id": scheme_id,
                    "model_id": model_id,
                    "hidden_state_index": metric_hidden_state,
                    "evaluation_iteration": evaluation_iteration,
                    "n_observations": len(subset),
                    "n_groups": subset["evaluation_group_id"].nunique(),
                    **_metric_values(
                        subset["y_true"].to_numpy(dtype=float),
                        subset["y_pred"].to_numpy(dtype=float),
                    ),
                }
            )

    return pd.DataFrame(rows)


def prepare_esm2_evaluation(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (
        observations,
        schemes,
        assignments,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    ) = _load_inputs(repo_root)

    _validate_inputs(
        observations,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    _validate_embedding_record(
        repo_root,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )

    selected_schemes = schemes[schemes["scheme_id"].isin(CORE_SCHEME_IDS)].copy()

    if set(selected_schemes["scheme_id"]) != set(CORE_SCHEME_IDS):
        raise SequenceModelSensitivityError("Core evaluation schemes are incomplete")

    if not selected_schemes["status"].eq("ready").all():
        raise SequenceModelSensitivityError("Core evaluation schemes must all be ready")

    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []

    for _, scheme in selected_schemes.sort_values("scheme_id").iterrows():
        frame = _join_scheme_observations(
            membership,
            observations,
            embedding_index,
            str(scheme["scheme_id"]),
        )

        family = str(scheme["assignment_family"])

        if family == "five_outer_folds_with_grouped_inner_validation":
            scheme_predictions, scheme_selection = _nested_evaluation(
                scheme,
                frame,
                assignments,
                mutant_embeddings,
                wild_type_embeddings,
            )

        elif family == "source_only_grouped_cv_then_target_test":
            scheme_predictions, scheme_selection = _source_cv_evaluation(
                scheme,
                frame,
                assignments,
                mutant_embeddings,
                wild_type_embeddings,
            )

        else:
            raise SequenceModelSensitivityError(
                f"Unsupported assignment family: {family}"
            )

        prediction_rows.extend(scheme_predictions)
        selection_rows.extend(scheme_selection)

    predictions = pd.DataFrame(prediction_rows)
    selection = pd.DataFrame(selection_rows)

    if predictions.empty:
        raise SequenceModelSensitivityError("No ESM-2 predictions were generated")

    metrics = _metric_rows(predictions)

    return predictions, metrics, selection


def write_esm2_evaluation(
    repo_root: Path,
) -> Mapping[str, object]:
    predictions, metrics, selection = prepare_esm2_evaluation(repo_root)

    output_dir = repo_root / "data/interim/sequence_model_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        output_dir / "esm2_predictions.csv",
        index=False,
    )
    metrics.to_csv(
        output_dir / "esm2_metrics.csv",
        index=False,
    )
    selection.to_csv(
        output_dir / "esm2_model_selection.csv",
        index=False,
    )

    summary = {
        "model_id": MODEL_ID,
        "candidate_hidden_state_indices": list(EMBEDDING_HIDDEN_STATE_INDICES),
        "candidate_alphas": list(ESM2_RIDGE_ALPHAS),
        "model_selection_metric": "mean_validation_mae",
        "model_selection_tie_break": (
            "lower hidden-state index, then lower alpha for exact mean-MAE ties"
        ),
        "scheme_ids": sorted(predictions["scheme_id"].unique().tolist()),
        "prediction_count": int(len(predictions)),
        "metric_row_count": int(len(metrics)),
        "model_selection_row_count": int(len(selection)),
        "test_labels_used_for_model_selection": False,
        "embedding_scaling": (
            "training-data mean and standard deviation within each "
            "model-selection or refit partition"
        ),
        "site_delta_definition": (
            "mutant-site ESM-2 hidden state minus the corresponding "
            "wild-type residue ESM-2 hidden state"
        ),
    }

    (output_dir / "esm2_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary


COMPARISON_BOOTSTRAP_REPLICATES = 2000
COMPARISON_BOOTSTRAP_SEED = "beta_lactamase_esm2_esmc_comparison"


def _metric_difference(metric: str, esm2_value: float, esmc_value: float) -> float:
    if metric in {"spearman", "pearson"}:
        return float(esm2_value - esmc_value)
    if metric in {"mae", "rmse"}:
        return float(esmc_value - esm2_value)
    raise SequenceModelSensitivityError(f"Unsupported comparison metric: {metric}")


def _comparison_seed(scheme_id: str, metric: str) -> int:
    value = f"{COMPARISON_BOOTSTRAP_SEED}|{scheme_id}|{metric}"
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _paired_bootstrap_difference(
    frame: pd.DataFrame,
    metric: str,
    *,
    replicates: int = COMPARISON_BOOTSTRAP_REPLICATES,
) -> tuple[float, float, int]:
    if replicates < 1:
        raise SequenceModelSensitivityError(
            "Bootstrap replicate count must be positive"
        )
    groups = frame["evaluation_group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan"), float("nan"), 0
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(
        _comparison_seed(str(frame["scheme_id"].iloc[0]), metric)
    )
    values: list[float] = []
    y_true_all = frame["y_true"].to_numpy(dtype=float)
    esm2_all = frame["esm2_prediction"].to_numpy(dtype=float)
    esmc_all = frame["esmc_prediction"].to_numpy(dtype=float)
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in sampled])
        esm2_metrics = _metric_values(y_true_all[indices], esm2_all[indices])
        esmc_metrics = _metric_values(y_true_all[indices], esmc_all[indices])
        value = _metric_difference(metric, esm2_metrics[metric], esmc_metrics[metric])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        int(len(array)),
    )


def compare_esm2_with_esmc(
    repo_root: Path,
    *,
    bootstrap_replicates: int = COMPARISON_BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = repo_root / "data/interim/sequence_model_sensitivity"
    esm2_path = output_dir / "esm2_predictions.csv"
    esmc_path = (
        repo_root / "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv"
    )
    if not esm2_path.is_file() or not esmc_path.is_file():
        raise SequenceModelSensitivityError(
            "ESM-2 or ESMC prediction records are missing"
        )
    dtype = {"evaluation_iteration": str, "observation_id": str}
    esm2 = pd.read_csv(esm2_path, dtype=dtype, low_memory=False)
    esmc = pd.read_csv(esmc_path, dtype=dtype, low_memory=False)
    esmc = esmc[esmc["model_id"].astype(str).eq("esmc_site_delta_ridge")].copy()
    key = [
        "scheme_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
    ]
    if esm2.duplicated(key).any() or esmc.duplicated(key).any():
        raise SequenceModelSensitivityError(
            "Sequence prediction identities are not unique"
        )
    left = esm2[
        key
        + [
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
            "y_pred",
        ]
    ].rename(columns={"y_pred": "esm2_prediction"})
    right = esmc[key + ["y_true", "y_pred"]].rename(
        columns={"y_true": "esmc_y_true", "y_pred": "esmc_prediction"}
    )
    matched = left.merge(
        right, on=key, how="outer", validate="one_to_one", indicator=True
    )
    if not matched["_merge"].eq("both").all():
        counts = matched["_merge"].value_counts().to_dict()
        raise SequenceModelSensitivityError(
            f"ESM-2 and ESMC evaluation populations differ: {counts}"
        )
    matched = matched.drop(columns="_merge")
    target_difference = np.abs(
        matched["y_true"].to_numpy(dtype=float)
        - matched["esmc_y_true"].to_numpy(dtype=float)
    )
    if (
        not np.isfinite(target_difference).all()
        or float(target_difference.max()) > 1e-9
    ):
        raise SequenceModelSensitivityError("ESM-2 and ESMC targets disagree")
    matched = matched.drop(columns="esmc_y_true")
    matched["esm2_absolute_error"] = np.abs(
        matched["y_true"].to_numpy(dtype=float)
        - matched["esm2_prediction"].to_numpy(dtype=float)
    )
    matched["esmc_absolute_error"] = np.abs(
        matched["y_true"].to_numpy(dtype=float)
        - matched["esmc_prediction"].to_numpy(dtype=float)
    )
    matched["esm2_minus_esmc_absolute_error"] = (
        matched["esm2_absolute_error"] - matched["esmc_absolute_error"]
    )
    rows: list[Mapping[str, object]] = []
    for scheme_id, part in matched.groupby("scheme_id", sort=True):
        esm2_metrics = _metric_values(
            part["y_true"].to_numpy(dtype=float),
            part["esm2_prediction"].to_numpy(dtype=float),
        )
        esmc_metrics = _metric_values(
            part["y_true"].to_numpy(dtype=float),
            part["esmc_prediction"].to_numpy(dtype=float),
        )
        for metric in ("spearman", "pearson", "mae", "rmse"):
            difference = _metric_difference(
                metric, esm2_metrics[metric], esmc_metrics[metric]
            )
            lower, upper, valid = _paired_bootstrap_difference(
                part,
                metric,
                replicates=bootstrap_replicates,
            )
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "model_a": MODEL_ID,
                    "model_b": "esmc_site_delta_ridge",
                    "metric": metric,
                    "model_a_value": esm2_metrics[metric],
                    "model_b_value": esmc_metrics[metric],
                    "paired_difference": difference,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_observations": len(part),
                    "n_groups": part["evaluation_group_id"].nunique(),
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_valid_replicates": valid,
                    "resampling_unit": "evaluation_group_id",
                    "difference_interpretation": "positive_favours_esm2",
                }
            )
    return matched, pd.DataFrame(rows)


def write_sequence_model_sensitivity(
    repo_root: Path,
    *,
    bootstrap_replicates: int = COMPARISON_BOOTSTRAP_REPLICATES,
) -> Mapping[str, object]:
    evaluation = write_esm2_evaluation(repo_root)
    matched, comparisons = compare_esm2_with_esmc(
        repo_root,
        bootstrap_replicates=bootstrap_replicates,
    )
    output_dir = repo_root / "data/interim/sequence_model_sensitivity"
    _write_csv(matched, output_dir / "esm2_esmc_matched_predictions.csv")
    _write_csv(comparisons, output_dir / "esm2_esmc_paired_comparisons.csv")
    extraction = json.loads(
        (output_dir / "esm2_embeddings/extraction_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "status": "PASS",
        "analysis_role": "sequence_model_sensitivity_check",
        "primary_sequence_model_id": "esmc_site_delta_ridge",
        "sensitivity_model_id": MODEL_ID,
        "sensitivity_source_model_id": ESM2_SOURCE_MODEL_ID,
        "sensitivity_source_model_revision": extraction["model_revision"],
        "candidate_hidden_state_indices": list(EMBEDDING_HIDDEN_STATE_INDICES),
        "candidate_alphas": list(ESM2_RIDGE_ALPHAS),
        "prediction_count": int(evaluation["prediction_count"]),
        "matched_prediction_count": int(len(matched)),
        "paired_comparison_count": int(len(comparisons)),
        "bootstrap_replicates": int(bootstrap_replicates),
        "model_selection_uses_only_permitted_validation_rows": True,
        "target_protein_labels_used_for_cross_protein_model_selection": False,
        "existing_esmc_predictions_changed": False,
        "existing_primary_results_changed": False,
        "outputs": [
            "esm2_predictions.csv",
            "esm2_metrics.csv",
            "esm2_model_selection.csv",
            "esm2_evaluation_summary.json",
            "esm2_esmc_matched_predictions.csv",
            "esm2_esmc_paired_comparisons.csv",
            "sequence_model_sensitivity_summary.json",
        ],
    }
    (output_dir / "sequence_model_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
