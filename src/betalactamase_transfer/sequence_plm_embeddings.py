from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .sequence_neighbourhood_evaluation import _load_assay_sequences

MODEL_ID = "biohub/ESMC-600M"
MODEL_REVISION = "a7e82012c83126b9eedb055fea9fa84b6c02f094"
MODEL_HIDDEN_SIZE = 1152
MODEL_HIDDEN_STATE_COUNT = 37
DEFAULT_HIDDEN_STATE_INDICES = (9, 18, 27, 36)
DEFAULT_BATCH_SIZE = 8
DEFAULT_PRECISION = "float16"

IDENTITY_COLUMNS = (
    "protein_id",
    "assay_mutation",
    "assay_position",
    "wild_type",
    "mutant",
    "sequence_task_eligible",
)


class SequencePLMEmbeddingError(RuntimeError):
    """Raised when a PLM embedding cannot be generated safely."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """Frozen runtime choices for ESMC embedding extraction."""

    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    hidden_state_indices: tuple[int, ...] = DEFAULT_HIDDEN_STATE_INDICES
    batch_size: int = DEFAULT_BATCH_SIZE
    precision: str = DEFAULT_PRECISION


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _embedding_index_digest(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n")
    return _sha256_text(payload)


def _validate_hidden_state_indices(indices: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in indices)
    if not values:
        raise SequencePLMEmbeddingError("At least one hidden-state index is required")
    if len(set(values)) != len(values):
        raise SequencePLMEmbeddingError("Hidden-state indices must be unique")
    if min(values) < 0 or max(values) >= MODEL_HIDDEN_STATE_COUNT:
        raise SequencePLMEmbeddingError(
            f"Hidden-state indices must be between 0 and {MODEL_HIDDEN_STATE_COUNT - 1}"
        )
    return values


def _make_mutant_sequence(
    wild_type_sequence: str,
    assay_position: int,
    expected_wild_type: str,
    mutant: str,
) -> str:
    position = int(assay_position)
    if position < 1 or position > len(wild_type_sequence):
        raise SequencePLMEmbeddingError(
            f"Assay position {position} is outside a "
            f"{len(wild_type_sequence)}-residue sequence"
        )
    observed = wild_type_sequence[position - 1]
    if observed != expected_wild_type:
        raise SequencePLMEmbeddingError(
            f"Wild-type mismatch at assay position {position}: "
            f"expected {expected_wild_type}, observed {observed}"
        )
    if mutant == expected_wild_type:
        raise SequencePLMEmbeddingError("Missense mutation does not change the residue")
    if len(expected_wild_type) != 1 or len(mutant) != 1:
        raise SequencePLMEmbeddingError("Residue identities must be one-letter codes")
    return wild_type_sequence[: position - 1] + mutant + wild_type_sequence[position:]


def build_mutation_embedding_index(repo_root: Path) -> pd.DataFrame:
    """Build the unique mutation-to-embedding index used for ESMC extraction."""
    path = repo_root / "data/processed/missense_observations.csv"
    if not path.is_file():
        raise SequencePLMEmbeddingError(f"Missing required input: {path}")

    frame = pd.read_csv(path, usecols=list(IDENTITY_COLUMNS), low_memory=False)
    if frame[list(IDENTITY_COLUMNS)].isna().any().any():
        raise SequencePLMEmbeddingError(
            "Mutation identity columns contain missing values"
        )
    if not frame["sequence_task_eligible"].astype(bool).all():
        raise SequencePLMEmbeddingError(
            "Unexpected sequence-ineligible rows in missense observations"
        )

    identity = frame[
        ["protein_id", "assay_mutation", "assay_position", "wild_type", "mutant"]
    ].drop_duplicates()

    duplicate_labels = identity.duplicated(
        subset=["protein_id", "assay_mutation"], keep=False
    )
    if duplicate_labels.any():
        raise SequencePLMEmbeddingError(
            "A protein-specific assay mutation maps to multiple residue identities"
        )

    sequences = _load_assay_sequences(repo_root)
    records = []
    for row in identity.itertuples(index=False):
        protein_id = str(row.protein_id)
        if protein_id not in sequences:
            raise SequencePLMEmbeddingError(f"Unknown protein: {protein_id}")
        position = int(row.assay_position)
        wild_type = str(row.wild_type)
        mutant = str(row.mutant)
        mutant_sequence = _make_mutant_sequence(
            sequences[protein_id], position, wild_type, mutant
        )
        expected_label = f"{wild_type}{position}{mutant}"
        if str(row.assay_mutation) != expected_label:
            raise SequencePLMEmbeddingError(
                f"Mutation label mismatch: {row.assay_mutation} != {expected_label}"
            )
        records.append(
            {
                "protein_id": protein_id,
                "assay_mutation": str(row.assay_mutation),
                "assay_position": position,
                "wild_type": wild_type,
                "mutant": mutant,
                "mutant_sequence_sha256": _sha256_text(mutant_sequence),
            }
        )

    embedding_index = pd.DataFrame(records).sort_values(
        ["protein_id", "assay_position", "mutant", "assay_mutation"],
        kind="mergesort",
    )
    embedding_index = embedding_index.reset_index(drop=True)
    embedding_index.insert(
        0, "embedding_row", np.arange(len(embedding_index), dtype=int)
    )

    if set(embedding_index["protein_id"]) != {"NDM-1", "VIM-2"}:
        raise SequencePLMEmbeddingError("Expected NDM-1 and VIM-2 mutations")
    return embedding_index


def prepare_sequence_plm_embedding_index(
    repo_root: Path,
    config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
) -> Mapping[str, object]:
    """Write the outcome-blind embedding_index and extraction contract."""
    hidden_state_indices = _validate_hidden_state_indices(config.hidden_state_indices)
    if config.batch_size <= 0:
        raise SequencePLMEmbeddingError("Batch size must be positive")
    if config.precision != "float16":
        raise SequencePLMEmbeddingError(
            "The verified production precision is currently float16"
        )

    embedding_index = build_mutation_embedding_index(repo_root)
    assay_sequences = _load_assay_sequences(repo_root)
    output_dir = repo_root / "data/interim/sequence_plm_embeddings"
    _write_csv(embedding_index, output_dir / "mutation_embedding_index.csv")

    summary = {
        "model_id": config.model_id,
        "model_revision": config.revision,
        "precision": config.precision,
        "batch_size": config.batch_size,
        "hidden_state_indices": list(hidden_state_indices),
        "hidden_size": MODEL_HIDDEN_SIZE,
        "mutation_count": len(embedding_index),
        "protein_counts": {
            protein_id: int(count)
            for protein_id, count in embedding_index.groupby("protein_id")
            .size()
            .items()
        },
        "assay_sequence_lengths": {
            protein_id: len(sequence)
            for protein_id, sequence in assay_sequences.items()
        },
        "assay_sequence_sha256": {
            protein_id: _sha256_text(sequence)
            for protein_id, sequence in assay_sequences.items()
        },
        "embedding_index_sha256": _embedding_index_digest(embedding_index),
        "outcome_columns_loaded": False,
        "embedding_definition": (
            "ESMC hidden states at the mutated assay-residue token for each mutant; "
            "wild-type residue states are extracted once per protein and "
            "mutation deltas are computed against the corresponding "
            "wild-type residue state"
        ),
        "primary_hidden_state_index": 36,
        "candidate_hidden_state_policy": (
            "quarter-depth hidden-state indices 9, 18, 27, and 36 are fixed "
            "before outcome-based model selection; final-test labels are not "
            "used to choose among them"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _torch_dtype(precision: str):
    try:
        import torch
    except ImportError as exc:
        raise SequencePLMEmbeddingError(
            "PyTorch is required for PLM extraction"
        ) from exc
    if precision == "float16":
        return torch.float16
    raise SequencePLMEmbeddingError(f"Unsupported precision: {precision}")


def extract_sequence_plm_embeddings(
    repo_root: Path,
    config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
    *,
    local_files_only: bool = True,
    overwrite: bool = False,
) -> Mapping[str, object]:
    """Extract mutation-site ESMC states for all unique missense substitutions."""
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SequencePLMEmbeddingError(
            "PLM extraction requires torch and transformers in the active environment"
        ) from exc

    if not torch.cuda.is_available():
        raise SequencePLMEmbeddingError(
            "CUDA is required for production PLM extraction"
        )

    hidden_state_indices = _validate_hidden_state_indices(config.hidden_state_indices)
    embedding_index = build_mutation_embedding_index(repo_root)
    assay_sequences = _load_assay_sequences(repo_root)
    output_dir = repo_root / "data/interim/sequence_plm_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "extraction_summary.json"
    mutant_path = output_dir / "mutant_site_embeddings.npy"
    wt_path = output_dir / "wild_type_residue_embeddings.npy"
    index_path = output_dir / "mutation_embedding_index.csv"

    mutant_tmp_path = output_dir / "mutant_site_embeddings.npy.tmp"
    wt_tmp_path = output_dir / "wild_type_residue_embeddings.npy.tmp"

    existing = [path for path in (summary_path, mutant_path, wt_path) if path.exists()]
    if existing and not overwrite:
        raise SequencePLMEmbeddingError(
            "PLM outputs already exist; pass overwrite=True only after "
            "verifying the existing run"
        )

    _write_csv(embedding_index, index_path)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        local_files_only=local_files_only,
    )
    dtype = _torch_dtype(config.precision)
    model = AutoModel.from_pretrained(
        config.model_id,
        revision=config.revision,
        dtype=dtype,
        local_files_only=local_files_only,
    ).to("cuda")
    model.eval()

    if int(model.config.d_model) != MODEL_HIDDEN_SIZE:
        raise SequencePLMEmbeddingError(
            f"Unexpected ESMC hidden size: {model.config.d_model}"
        )
    if int(model.config.n_layers) + 1 != MODEL_HIDDEN_STATE_COUNT:
        raise SequencePLMEmbeddingError(
            f"Unexpected ESMC hidden-state count: {int(model.config.n_layers) + 1}"
        )

    protein_order = ("NDM-1", "VIM-2")
    max_length = max(len(assay_sequences[protein]) for protein in protein_order)
    for temporary_path in (mutant_tmp_path, wt_tmp_path):
        if temporary_path.exists():
            temporary_path.unlink()

    wt_array = np.lib.format.open_memmap(
        wt_tmp_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(protein_order),
            len(hidden_state_indices),
            max_length,
            MODEL_HIDDEN_SIZE,
        ),
    )
    wt_array[:] = np.nan

    with torch.inference_mode():
        for protein_index, protein_id in enumerate(protein_order):
            sequence = assay_sequences[protein_id]
            encoded = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            if tokens[1:-1] != list(sequence):
                raise SequencePLMEmbeddingError(
                    f"Tokenizer residue contract failed for {protein_id}"
                )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            states = output.hidden_states
            if states is None or len(states) != MODEL_HIDDEN_STATE_COUNT:
                raise SequencePLMEmbeddingError(
                    f"Unexpected hidden-state output for {protein_id}"
                )
            for layer_index, hidden_state_index in enumerate(hidden_state_indices):
                residue_states = states[hidden_state_index][0, 1 : len(sequence) + 1, :]
                if not bool(torch.isfinite(residue_states).all()):
                    raise SequencePLMEmbeddingError(
                        f"Non-finite WT embedding for {protein_id}"
                    )
                wt_array[protein_index, layer_index, : len(sequence), :] = (
                    residue_states.detach().cpu().numpy().astype(np.float16, copy=False)
                )
    wt_array.flush()

    mutant_array = np.lib.format.open_memmap(
        mutant_tmp_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(embedding_index), len(hidden_state_indices), MODEL_HIDDEN_SIZE),
    )

    peak_allocated = 0
    with torch.inference_mode():
        for start in range(0, len(embedding_index), config.batch_size):
            batch = embedding_index.iloc[start : start + config.batch_size]
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
                tokenizer.convert_ids_to_tokens(encoded["input_ids"][index])
                for index in range(len(batch))
            ]
            for index, row in enumerate(batch.itertuples(index=False)):
                position = int(row.assay_position)
                if token_rows[index][position] != str(row.mutant):
                    raise SequencePLMEmbeddingError(
                        "Mutant token mismatch for "
                        f"{row.protein_id}/{row.assay_mutation}"
                    )

            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            states = output.hidden_states
            if states is None or len(states) != MODEL_HIDDEN_STATE_COUNT:
                raise SequencePLMEmbeddingError("Unexpected mutant hidden-state output")

            for local_index, position in enumerate(positions):
                output_index = start + local_index
                for layer_index, hidden_state_index in enumerate(hidden_state_indices):
                    vector = states[hidden_state_index][local_index, position, :]
                    if not bool(torch.isfinite(vector).all()):
                        row = batch.iloc[local_index]
                        raise SequencePLMEmbeddingError(
                            "Non-finite mutant embedding for "
                            f"{row['protein_id']}/{row['assay_mutation']}"
                        )
                    mutant_array[output_index, layer_index, :] = (
                        vector.detach().cpu().numpy().astype(np.float16, copy=False)
                    )
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))

    mutant_array.flush()

    if not np.isfinite(mutant_array).all():
        raise SequencePLMEmbeddingError(
            "Mutant embedding array contains non-finite values"
        )
    for protein_index, protein_id in enumerate(protein_order):
        length = len(assay_sequences[protein_id])
        if not np.isfinite(wt_array[protein_index, :, :length, :]).all():
            raise SequencePLMEmbeddingError(
                f"WT embedding array contains non-finite values for {protein_id}"
            )

    summary = {
        "model_id": config.model_id,
        "model_revision": config.revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "precision": config.precision,
        "batch_size": config.batch_size,
        "hidden_state_indices": list(hidden_state_indices),
        "hidden_size": MODEL_HIDDEN_SIZE,
        "mutation_count": len(embedding_index),
        "embedding_index_sha256": _embedding_index_digest(embedding_index),
        "mutant_embedding_shape": list(mutant_array.shape),
        "wild_type_embedding_shape": list(wt_array.shape),
        "peak_allocated_bytes": peak_allocated,
        "outcome_columns_loaded": False,
        "token_index_contract": "one_based_assay_position_equals_token_index",
    }

    del mutant_array
    del wt_array

    mutant_tmp_path.replace(mutant_path)
    wt_tmp_path.replace(wt_path)

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
