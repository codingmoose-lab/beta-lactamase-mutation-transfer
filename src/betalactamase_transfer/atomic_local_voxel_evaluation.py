from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

import betalactamase_transfer.residue_graph_neural_evaluation as stage32


class AtomicLocalVoxelEvaluationError(RuntimeError):
    pass


MODEL_ID = "atomic_local_voxel_cnn"
REFERENCE_MODEL_IDS = (
    "centre_residue_neural_regressor",
    "residue_graph_neural_regressor",
    "egnn_residue_regressor",
    "experimental_structure_descriptor_ridge",
)
CORE_SCHEME_IDS = tuple(stage32.CORE_SCHEME_IDS)
TARGET_COLUMN = stage32.TARGET_COLUMN

VOXEL_CHANNELS = ("carbon", "nitrogen", "oxygen", "sulfur", "other", "centre")
VOXEL_SIZE_ANGSTROM = 1.0
VOXEL_HALF_WIDTH_ANGSTROM = 10.0
VOXEL_DIMENSION = 20

CNN_WIDTHS = (16, 32, 64)
READOUT_WIDTH = 128
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_NORM_LIMIT = 5.0
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 16384
TRAINING_REPEATS = 3
SELECTION_SEED = 43001
FINAL_REPEAT_SEEDS = (43101, 43102, 43103)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_atomic_local_voxel_bootstrap"
CHECKPOINT_DIRECTORY_NAME = "context_checkpoints"
CHECKPOINT_RECORD_NAME = "checkpoint_run.json"
TRAINING_COMPLETE_RECORD_NAME = "training_complete.json"

EXPECTED_GRAPH_NODE_COUNT = 469
EXPECTED_PREDICTIONS_PER_MODEL = 118204
EXPECTED_CONTEXT_COUNT = 17
EXPECTED_STRUCTURE_PROTEINS = ("NDM-1", "VIM-2")

COMPARISON_KEY = tuple(stage32.COMPARISON_KEY)
MATCHED_FIELDS = tuple(stage32.MATCHED_FIELDS)


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


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _normalised_position(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "?", "."}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _normalised_insertion(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "?", "."} else text


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_reproducible_execution() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _parse_protein_atoms(cif_path: Path, chain_id: str) -> pd.DataFrame:
    cif = MMCIF2Dict(str(cif_path))
    required = (
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
    )
    missing = [key for key in required if key not in cif]
    if missing:
        raise AtomicLocalVoxelEvaluationError(
            f"Missing atom-site fields in {cif_path}: {missing}"
        )
    columns = {key: _as_list(cif[key]) for key in required}
    count = len(columns["_atom_site.id"])
    optional = {
        "_atom_site.label_alt_id": ".",
        "_atom_site.pdbx_PDB_ins_code": "?",
        "_atom_site.pdbx_PDB_model_num": "1",
    }
    for key, default in optional.items():
        columns[key] = _as_list(cif[key]) if key in cif else [default] * count

    rows: list[dict[str, object]] = []
    for index in range(count):
        if columns["_atom_site.pdbx_PDB_model_num"][index] not in {
            "1",
            "1.0",
            ".",
            "?",
        }:
            continue
        if columns["_atom_site.group_PDB"][index] != "ATOM":
            continue
        if columns["_atom_site.auth_asym_id"][index] != chain_id:
            continue
        element = columns["_atom_site.type_symbol"][index].strip().upper()
        if element in {"H", "D"}:
            continue
        rows.append(
            {
                "atom_id": str(columns["_atom_site.id"][index]),
                "element": element,
                "atom_name": str(columns["_atom_site.label_atom_id"][index]).strip(),
                "author_position": _normalised_position(
                    columns["_atom_site.auth_seq_id"][index]
                ),
                "insertion_code": _normalised_insertion(
                    columns["_atom_site.pdbx_PDB_ins_code"][index]
                ),
                "altloc": _normalised_insertion(
                    columns["_atom_site.label_alt_id"][index]
                ),
                "occupancy": float(columns["_atom_site.occupancy"][index]),
                "x": float(columns["_atom_site.Cartn_x"][index]),
                "y": float(columns["_atom_site.Cartn_y"][index]),
                "z": float(columns["_atom_site.Cartn_z"][index]),
            }
        )
    if not rows:
        raise AtomicLocalVoxelEvaluationError(
            f"No model-1 protein heavy atoms found for chain {chain_id} in {cif_path}"
        )
    atoms = pd.DataFrame(rows)
    atoms["altloc_priority"] = atoms["altloc"].map(
        lambda value: 0 if value in {"", "A"} else 1
    )
    atoms = atoms.sort_values(
        [
            "author_position",
            "insertion_code",
            "atom_name",
            "occupancy",
            "altloc_priority",
            "atom_id",
        ],
        ascending=[True, True, True, False, True, True],
        kind="stable",
    )
    atoms = atoms.drop_duplicates(
        ["author_position", "insertion_code", "atom_name"], keep="first"
    ).drop(columns="altloc_priority")
    return atoms.reset_index(drop=True)


def _residue_local_frame(residue_atoms: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    coordinates: dict[str, np.ndarray] = {}
    for atom_name in ("N", "CA", "C"):
        rows = residue_atoms.loc[residue_atoms["atom_name"] == atom_name]
        if len(rows) != 1:
            raise AtomicLocalVoxelEvaluationError(
                f"Backbone atom {atom_name} is not uniquely available"
            )
        coordinates[atom_name] = rows[["x", "y", "z"]].to_numpy(dtype=float)[0]
    origin = coordinates["CA"]
    x_axis = coordinates["C"] - origin
    x_norm = float(np.linalg.norm(x_axis))
    if not math.isfinite(x_norm) or x_norm <= 1e-8:
        raise AtomicLocalVoxelEvaluationError("Cannot define local x-axis from CA-C")
    x_axis /= x_norm
    n_direction = coordinates["N"] - origin
    z_axis = np.cross(x_axis, n_direction)
    z_norm = float(np.linalg.norm(z_axis))
    if not math.isfinite(z_norm) or z_norm <= 1e-8:
        raise AtomicLocalVoxelEvaluationError("Cannot define local residue plane")
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.stack((x_axis, y_axis, z_axis), axis=1)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise AtomicLocalVoxelEvaluationError(
            "Local coordinate frame is not orthonormal"
        )
    if np.linalg.det(rotation) < 0.999:
        raise AtomicLocalVoxelEvaluationError(
            "Local coordinate frame is not right-handed"
        )
    return origin, rotation


def _element_channel(element: str) -> int:
    return {"C": 0, "N": 1, "O": 2, "S": 3}.get(element.upper(), 4)


def _deposit_trilinear(grid: np.ndarray, channel: int, coordinate: np.ndarray) -> None:
    continuous = (coordinate + VOXEL_HALF_WIDTH_ANGSTROM) / VOXEL_SIZE_ANGSTROM - 0.5
    base = np.floor(continuous).astype(int)
    fraction = continuous - base
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                index = base + np.array([dx, dy, dz], dtype=int)
                if np.any(index < 0) or np.any(index >= VOXEL_DIMENSION):
                    continue
                weight = (
                    (fraction[0] if dx else 1.0 - fraction[0])
                    * (fraction[1] if dy else 1.0 - fraction[1])
                    * (fraction[2] if dz else 1.0 - fraction[2])
                )
                grid[channel, index[0], index[1], index[2]] += float(weight)


def _voxelise_residue_environment(
    atoms: pd.DataFrame,
    author_position: str,
    insertion_code: str,
) -> np.ndarray:
    residue_mask = atoms["author_position"].eq(author_position) & atoms[
        "insertion_code"
    ].eq(insertion_code)
    residue_atoms = atoms.loc[residue_mask].copy()
    if residue_atoms.empty:
        raise AtomicLocalVoxelEvaluationError(
            f"No protein atoms found for author residue "
            f"{author_position}{insertion_code}"
        )
    origin, rotation = _residue_local_frame(residue_atoms)
    all_coordinates = atoms[["x", "y", "z"]].to_numpy(dtype=float)
    local = (all_coordinates - origin) @ rotation
    cube_mask = np.all(
        (local >= -VOXEL_HALF_WIDTH_ANGSTROM) & (local < VOXEL_HALF_WIDTH_ANGSTROM),
        axis=1,
    )
    selected = atoms.loc[cube_mask].reset_index(drop=True)
    selected_local = local[cube_mask]
    grid = np.zeros(
        (len(VOXEL_CHANNELS), VOXEL_DIMENSION, VOXEL_DIMENSION, VOXEL_DIMENSION),
        dtype=np.float32,
    )
    for row_index, atom in selected.iterrows():
        coordinate = selected_local[row_index]
        _deposit_trilinear(grid, _element_channel(str(atom["element"])), coordinate)
        if (
            str(atom["author_position"]) == author_position
            and str(atom["insertion_code"]) == insertion_code
        ):
            _deposit_trilinear(grid, 5, coordinate)
    if float(grid[:5].sum()) <= 0.0 or float(grid[5].sum()) <= 0.0:
        raise AtomicLocalVoxelEvaluationError("Voxel grid is unexpectedly empty")
    return grid


def _load_atomic_context(
    repo_root: Path,
) -> tuple[torch.Tensor, Mapping[tuple[str, int], int], pd.DataFrame]:
    nodes_path = repo_root / "data/interim/residue_graphs/residue_graph_nodes.csv"
    status_path = (
        repo_root / "data/interim/protein_structures/residue_coordinate_status.csv"
    )
    selection_path = (
        repo_root / "data/interim/protein_structures/selected_structures.json"
    )
    for path in (nodes_path, status_path, selection_path):
        if not path.is_file():
            raise AtomicLocalVoxelEvaluationError(f"Missing required input: {path}")
    nodes = pd.read_csv(nodes_path, low_memory=False)
    status = pd.read_csv(status_path, low_memory=False)
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    if len(nodes) != EXPECTED_GRAPH_NODE_COUNT:
        raise AtomicLocalVoxelEvaluationError(
            f"Unexpected residue graph node count: {len(nodes)}"
        )
    if tuple(sorted(selections)) != tuple(sorted(EXPECTED_STRUCTURE_PROTEINS)):
        raise AtomicLocalVoxelEvaluationError("Unexpected selected-structure proteins")

    grids: list[np.ndarray] = []
    lookup: dict[tuple[str, int], int] = {}
    audit_rows: list[dict[str, object]] = []
    for protein_id in EXPECTED_STRUCTURE_PROTEINS:
        selection = selections[protein_id]
        pdb_id = str(selection["pdb_id"])
        chain_id = str(selection["primary_chain"])
        cif_path = repo_root / f"data/raw/pdb/{pdb_id}/{pdb_id}.cif"
        if not cif_path.is_file():
            raise AtomicLocalVoxelEvaluationError(
                f"Missing selected PDB file: {cif_path}"
            )
        atoms = _parse_protein_atoms(cif_path, chain_id)
        protein_nodes = nodes.loc[nodes["protein_id"].astype(str) == protein_id].copy()
        protein_status = status.loc[
            (status["protein_id"].astype(str) == protein_id)
            & (status["pdb_id"].astype(str) == pdb_id)
            & (status["chain_id"].astype(str) == chain_id)
        ].copy()
        if protein_status.duplicated(["assay_position"]).any():
            raise AtomicLocalVoxelEvaluationError(
                "Selected-chain residue status is ambiguous"
            )
        status_lookup = protein_status.set_index("assay_position")
        for node in protein_nodes.sort_values("node_index", kind="stable").itertuples(
            index=False
        ):
            assay_position = int(node.assay_position)
            if assay_position not in status_lookup.index:
                raise AtomicLocalVoxelEvaluationError(
                    f"Missing coordinate status for {protein_id} "
                    f"position {assay_position}"
                )
            row = status_lookup.loc[assay_position]
            if not bool(row["coordinate_present"]) or not bool(
                row["backbone_complete"]
            ):
                raise AtomicLocalVoxelEvaluationError(
                    f"Voxel centre lacks complete backbone: {protein_id} "
                    f"{assay_position}"
                )
            author_position = _normalised_position(row["pdb_author_position"])
            insertion_code = _normalised_insertion(row["pdb_insertion_code"])
            grid = _voxelise_residue_environment(atoms, author_position, insertion_code)
            global_index = len(grids)
            key = (protein_id, assay_position)
            if key in lookup:
                raise AtomicLocalVoxelEvaluationError(
                    f"Duplicate voxel lookup key: {key}"
                )
            lookup[key] = global_index
            grids.append(grid)
            audit_rows.append(
                {
                    "protein_id": protein_id,
                    "assay_position": assay_position,
                    "node_index": int(node.node_index),
                    "pdb_id": pdb_id,
                    "pdb_chain": chain_id,
                    "pdb_author_position": author_position,
                    "pdb_insertion_code": insertion_code,
                    "voxel_index": global_index,
                    "protein_atom_density": float(grid[:5].sum()),
                    "centre_atom_density": float(grid[5].sum()),
                }
            )
    bank = np.stack(grids, axis=0)
    if bank.shape != (
        EXPECTED_GRAPH_NODE_COUNT,
        len(VOXEL_CHANNELS),
        VOXEL_DIMENSION,
        VOXEL_DIMENSION,
        VOXEL_DIMENSION,
    ):
        raise AtomicLocalVoxelEvaluationError(
            f"Unexpected voxel-bank shape: {bank.shape}"
        )
    if not np.isfinite(bank).all():
        raise AtomicLocalVoxelEvaluationError("Voxel bank contains non-finite values")
    return torch.from_numpy(bank), lookup, pd.DataFrame(audit_rows)


class NonOverlappingMaxPool3d(nn.Module):
    """Apply non-overlapping 2 x 2 x 2 maximum reduction."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            raise AtomicLocalVoxelEvaluationError(
                "Maximum reduction expects a five-dimensional tensor"
            )
        _, _, depth, height, width = values.shape
        if depth % 2 or height % 2 or width % 2:
            raise AtomicLocalVoxelEvaluationError(
                "Maximum reduction requires even depth, height, and width"
            )

        batch, channels = values.shape[:2]
        reduced = values.reshape(
            batch,
            channels,
            depth // 2,
            2,
            height // 2,
            2,
            width // 2,
            2,
        )
        reduced = torch.max(reduced, dim=7).values
        reduced = torch.max(reduced, dim=5).values
        reduced = torch.max(reduced, dim=3).values
        return reduced


class AtomicLocalVoxelCNN(nn.Module):
    def __init__(self, condition_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(len(VOXEL_CHANNELS), CNN_WIDTHS[0], kernel_size=3, padding=1),
            nn.ReLU(),
            NonOverlappingMaxPool3d(),
            nn.Conv3d(CNN_WIDTHS[0], CNN_WIDTHS[1], kernel_size=3, padding=1),
            nn.ReLU(),
            NonOverlappingMaxPool3d(),
            nn.Conv3d(CNN_WIDTHS[1], CNN_WIDTHS[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        readout_dim = CNN_WIDTHS[2] + len(stage32.SUBSTITUTIONS) + condition_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, READOUT_WIDTH),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(READOUT_WIDTH, READOUT_WIDTH // 2),
            nn.ReLU(),
            nn.Linear(READOUT_WIDTH // 2, 1),
        )

    def encode_voxel_bank(self, voxel_bank: torch.Tensor) -> torch.Tensor:
        return self.encoder(voxel_bank).flatten(1)

    def predict_from_encoding(
        self,
        encoded_bank: torch.Tensor,
        voxel_index: torch.Tensor,
        substitution: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            (encoded_bank[voxel_index], substitution, condition), dim=1
        )
        return self.readout(combined).squeeze(1)

    def forward(
        self,
        voxel_bank: torch.Tensor,
        voxel_index: torch.Tensor,
        substitution: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        encoded_bank = self.encode_voxel_bank(voxel_bank)
        return self.predict_from_encoding(
            encoded_bank, voxel_index, substitution, condition
        )


def _voxel_indices(
    frame: pd.DataFrame, lookup: Mapping[tuple[str, int], int]
) -> np.ndarray:
    indices: list[int] = []
    for row in frame.itertuples(index=False):
        key = (str(row.protein_id), int(row.assay_position))
        if key not in lookup:
            raise AtomicLocalVoxelEvaluationError(
                f"No voxel grid for observation key: {key}"
            )
        indices.append(int(lookup[key]))
    return np.asarray(indices, dtype=np.int64)


def _prepare_arrays(
    frame: pd.DataFrame,
    lookup: Mapping[tuple[str, int], int],
    condition_transform: stage32.ConditionTransform,
    target_mean: float,
    target_sd: float,
) -> Mapping[str, np.ndarray]:
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise AtomicLocalVoxelEvaluationError("Observation outcomes are not finite")
    return {
        "voxel_index": _voxel_indices(frame, lookup),
        "substitution": stage32._substitution_matrix(frame),
        "condition": condition_transform.encode(frame),
        "target": ((target - target_mean) / target_sd).astype(np.float32),
        "target_original": target.astype(np.float64),
    }


def _arrays_to_device(
    arrays: Mapping[str, np.ndarray], device: torch.device
) -> Mapping[str, torch.Tensor]:
    return {
        "voxel_index": torch.from_numpy(arrays["voxel_index"]).long().to(device),
        "substitution": torch.from_numpy(arrays["substitution"]).float().to(device),
        "condition": torch.from_numpy(arrays["condition"]).float().to(device),
        "target": torch.from_numpy(arrays["target"]).float().to(device),
    }


def _predict_standardised(
    model: AtomicLocalVoxelCNN,
    arrays: Mapping[str, np.ndarray],
    voxel_bank: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    encoded_bank = model.encode_voxel_bank(voxel_bank)
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays["voxel_index"]), batch_size):
            stop = min(start + batch_size, len(arrays["voxel_index"]))
            voxel_index = (
                torch.from_numpy(arrays["voxel_index"][start:stop]).long().to(device)
            )
            substitution = (
                torch.from_numpy(arrays["substitution"][start:stop]).float().to(device)
            )
            condition = (
                torch.from_numpy(arrays["condition"][start:stop]).float().to(device)
            )
            prediction = model.predict_from_encoding(
                encoded_bank, voxel_index, substitution, condition
            )
            result.append(prediction.detach().cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def _fit_with_validation(
    fitting_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    voxel_bank: torch.Tensor,
    lookup: Mapping[tuple[str, int], int],
    condition_mode: str,
    device: torch.device,
    seed: int,
    description: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[int, float, list[Mapping[str, object]]]:
    _set_seed(seed)
    condition_transform = stage32._fit_condition_transform(fitting_rows, condition_mode)
    target_mean, target_sd = stage32._target_transform(fitting_rows)
    fitting_arrays = _prepare_arrays(
        fitting_rows, lookup, condition_transform, target_mean, target_sd
    )
    validation_arrays = _prepare_arrays(
        validation_rows, lookup, condition_transform, target_mean, target_sd
    )
    model = AtomicLocalVoxelCNN(condition_transform.output_dimension).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.MSELoss()
    fitting_device = _arrays_to_device(fitting_arrays, device)
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
        optimiser.zero_grad(set_to_none=True)
        prediction = model(
            voxel_bank,
            fitting_device["voxel_index"],
            fitting_device["substitution"],
            fitting_device["condition"],
        )
        loss = loss_function(prediction, fitting_device["target"])
        loss.backward()
        clip_grad_norm_(model.parameters(), GRADIENT_NORM_LIMIT)
        optimiser.step()
        training_loss = float(loss.detach().cpu())
        validation_standardised = _predict_standardised(
            model, validation_arrays, voxel_bank, device, batch_size
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
        raise AtomicLocalVoxelEvaluationError("Voxel-CNN epoch selection failed")
    return best_epoch, best_validation_mae, history


def _fit_fixed_epochs_and_predict(
    fitting_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    voxel_bank: torch.Tensor,
    lookup: Mapping[tuple[str, int], int],
    condition_mode: str,
    device: torch.device,
    seed: int,
    epochs: int,
    description: str,
    batch_size: int,
) -> tuple[np.ndarray, list[Mapping[str, object]]]:
    _set_seed(seed)
    condition_transform = stage32._fit_condition_transform(fitting_rows, condition_mode)
    target_mean, target_sd = stage32._target_transform(fitting_rows)
    fitting_arrays = _prepare_arrays(
        fitting_rows, lookup, condition_transform, target_mean, target_sd
    )
    test_arrays = _prepare_arrays(
        test_rows, lookup, condition_transform, target_mean, target_sd
    )
    model = AtomicLocalVoxelCNN(condition_transform.output_dimension).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.MSELoss()
    fitting_device = _arrays_to_device(fitting_arrays, device)
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
        optimiser.zero_grad(set_to_none=True)
        prediction = model(
            voxel_bank,
            fitting_device["voxel_index"],
            fitting_device["substitution"],
            fitting_device["condition"],
        )
        loss = loss_function(prediction, fitting_device["target"])
        loss.backward()
        clip_grad_norm_(model.parameters(), GRADIENT_NORM_LIMIT)
        optimiser.step()
        training_loss = float(loss.detach().cpu())
        history.append({"epoch": epoch, "training_loss": training_loss})
        progress.set_postfix(train_loss=f"{training_loss:.4f}", refresh=False)
    standardised = _predict_standardised(
        model, test_arrays, voxel_bank, device, batch_size
    )
    prediction = standardised * target_sd + target_mean
    if not np.isfinite(prediction).all():
        raise AtomicLocalVoxelEvaluationError("Voxel-CNN predictions are not finite")
    return prediction, history


def _selection_for_context(
    context: Mapping[str, object],
    voxel_bank: torch.Tensor,
    lookup: Mapping[tuple[str, int], int],
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[int, list[Mapping[str, object]]]:
    scheme_id = str(context["scheme_id"])
    condition_mode = stage32.CONDITION_MODE_BY_SCHEME[scheme_id]
    history_rows: list[Mapping[str, object]] = []
    if "source_folds" not in context:
        epoch_count, validation_mae, history = _fit_with_validation(
            context["train"],
            context["validation"],
            voxel_bank,
            lookup,
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
            voxel_bank,
            lookup,
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
    return (
        context["target"].copy()
        if "source_folds" in context
        else context["test"].copy()
    )


def _fitting_rows(context: Mapping[str, object]) -> pd.DataFrame:
    if "source_folds" in context:
        return context["source"].copy()
    return pd.concat(
        [context["train"], context["validation"]], ignore_index=True
    ).drop_duplicates("observation_id")


def _prediction_records(
    context: Mapping[str, object],
    test_rows: pd.DataFrame,
    predictions: np.ndarray,
    repeat_index: int,
) -> list[Mapping[str, object]]:
    reference = context["reference"].copy().reset_index(drop=True)
    ordered_test = reference[["observation_id"]].merge(
        test_rows, on="observation_id", how="left", validate="one_to_one"
    )
    if ordered_test[TARGET_COLUMN].isna().any() or len(predictions) != len(
        ordered_test
    ):
        raise AtomicLocalVoxelEvaluationError("Unable to align voxel-CNN predictions")
    rows: list[Mapping[str, object]] = []
    for row_index, reference_row in reference.iterrows():
        record = {
            column: reference_row[column]
            for column in (*COMPARISON_KEY, *MATCHED_FIELDS)
        }
        record["model_id"] = MODEL_ID
        record["training_repeat"] = repeat_index
        record["y_pred"] = float(predictions[row_index])
        rows.append(record)
    return rows


def _normalise_evaluation_iteration(frame: pd.DataFrame) -> pd.DataFrame:
    if "evaluation_iteration" not in frame.columns:
        raise AtomicLocalVoxelEvaluationError(
            "Prediction table is missing evaluation_iteration"
        )

    normalised = frame.copy()

    if normalised["evaluation_iteration"].isna().any():
        raise AtomicLocalVoxelEvaluationError(
            "Prediction table contains missing evaluation_iteration values"
        )

    normalised["evaluation_iteration"] = (
        normalised["evaluation_iteration"].astype(str).str.strip()
    )

    if normalised["evaluation_iteration"].eq("").any():
        raise AtomicLocalVoxelEvaluationError(
            "Prediction table contains empty evaluation_iteration values"
        )

    return normalised


def _average_repeat_predictions(repeat_frame: pd.DataFrame) -> pd.DataFrame:
    normalised = _normalise_evaluation_iteration(repeat_frame)
    identity = [*COMPARISON_KEY, *MATCHED_FIELDS, "model_id"]
    averaged = normalised.groupby(identity, as_index=False, dropna=False)[
        "y_pred"
    ].mean()
    return averaged


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
    repo_root: Path,
) -> Mapping[str, object]:
    return {
        "model_id": MODEL_ID,
        "repeats": repeats,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "selection_seed": SELECTION_SEED,
        "final_repeat_seeds": list(FINAL_REPEAT_SEEDS[:repeats]),
        "voxel_channels": list(VOXEL_CHANNELS),
        "voxel_size_angstrom": VOXEL_SIZE_ANGSTROM,
        "voxel_half_width_angstrom": VOXEL_HALF_WIDTH_ANGSTROM,
        "voxel_dimension": VOXEL_DIMENSION,
        "coordinate_frame": "backbone_N_CA_C_right_handed",
        "atom_scope": "selected_experimental_PDB_primary_chain_ATOM_heavy_atoms_only",
        "hetero_atoms_included": False,
        "predicted_mutant_structures_used": False,
        "source_sha256": _sha256(Path(__file__)),
        "pdb_sha256": {
            "3SPU": _sha256(repo_root / "data/raw/pdb/3SPU/3SPU.cif"),
            "5YD7": _sha256(repo_root / "data/raw/pdb/5YD7/5YD7.cif"),
        },
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    }


def _checkpoint_paths(
    checkpoint_dir: Path, context: Mapping[str, object]
) -> tuple[Path, Path, Path]:
    token = _context_token(context)
    return (
        checkpoint_dir / f"{token}__repeat_predictions.csv",
        checkpoint_dir / f"{token}__training_history.csv",
        checkpoint_dir / f"{token}__complete.json",
    )


def _prepare_checkpoint_directory(
    output_dir: Path,
    contract: Mapping[str, object],
    *,
    postprocess_only: bool = False,
) -> Path:
    checkpoint_dir = output_dir / CHECKPOINT_DIRECTORY_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record_path = checkpoint_dir / CHECKPOINT_RECORD_NAME
    expected = dict(contract)

    if record_path.exists():
        recorded = json.loads(record_path.read_text(encoding="utf-8"))

        if recorded != expected:
            if not postprocess_only:
                raise AtomicLocalVoxelEvaluationError(
                    "Existing atomic voxel checkpoints use different execution settings"
                )

            recorded_without_source = {
                key: value for key, value in recorded.items() if key != "source_sha256"
            }
            expected_without_source = {
                key: value for key, value in expected.items() if key != "source_sha256"
            }

            if recorded_without_source != expected_without_source:
                raise AtomicLocalVoxelEvaluationError(
                    "Completed atomic voxel checkpoints use different training settings"
                )
    elif postprocess_only:
        raise AtomicLocalVoxelEvaluationError(
            "Post-processing requires the completed training run record"
        )
    else:
        _write_json(expected, record_path)

    return checkpoint_dir


def _write_context_checkpoint(
    checkpoint_dir: Path,
    context: Mapping[str, object],
    repeat_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
    selected_epoch: int,
    test_rows: pd.DataFrame,
    repeats: int,
) -> None:
    repeat_path, history_path, complete_path = _checkpoint_paths(
        checkpoint_dir, context
    )
    expected = len(test_rows) * repeats
    if len(repeat_frame) != expected:
        raise AtomicLocalVoxelEvaluationError(
            "Context checkpoint has an unexpected prediction count"
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
    repeat_path, history_path, complete_path = _checkpoint_paths(
        checkpoint_dir, context
    )
    if not complete_path.exists():
        return None
    if not repeat_path.exists() or not history_path.exists():
        raise AtomicLocalVoxelEvaluationError(
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
            raise AtomicLocalVoxelEvaluationError(
                f"Context checkpoint contract mismatch: {key}"
            )
    if record.get("repeat_predictions_sha256") != _sha256(repeat_path) or record.get(
        "training_history_sha256"
    ) != _sha256(history_path):
        raise AtomicLocalVoxelEvaluationError("Context checkpoint hash mismatch")
    return (
        pd.read_csv(repeat_path, low_memory=False),
        pd.read_csv(history_path, low_memory=False),
        int(record["selected_epoch_count"]),
    )


def _load_comparator_predictions(repo_root: Path) -> pd.DataFrame:
    paths = {
        "centre_and_graph": repo_root / "data/interim/residue_graph_neural_evaluation/"
        "residue_graph_neural_predictions.csv",
        "egnn": repo_root / "data/interim/residue_geometry_neural_evaluation/"
        "residue_geometry_neural_predictions.csv",
        "structure": repo_root / "data/interim/experimental_structure_evaluation/"
        "experimental_structure_predictions.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise AtomicLocalVoxelEvaluationError(
                f"Missing comparator predictions: {path}"
            )
    frames: list[pd.DataFrame] = []
    for path in paths.values():
        frame = pd.read_csv(path, low_memory=False)
        frames.append(
            frame.loc[frame["model_id"].astype(str).isin(REFERENCE_MODEL_IDS)].copy()
        )
    combined = _normalise_evaluation_iteration(pd.concat(frames, ignore_index=True))
    combined = combined.drop_duplicates([*COMPARISON_KEY, "model_id"], keep="first")
    counts = combined["model_id"].value_counts().to_dict()
    if set(counts) != set(REFERENCE_MODEL_IDS):
        raise AtomicLocalVoxelEvaluationError(
            f"Comparator model set is incomplete: {counts}"
        )
    if any(
        int(counts[model_id]) != EXPECTED_PREDICTIONS_PER_MODEL
        for model_id in REFERENCE_MODEL_IDS
    ):
        raise AtomicLocalVoxelEvaluationError(
            "Comparator prediction populations are incomplete"
        )
    return combined


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme_id, model_id, iteration), group in predictions.groupby(
        ["scheme_id", "model_id", "evaluation_iteration"],
        sort=True,
        dropna=False,
    ):
        values = stage32._metric_values(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "scheme_id": scheme_id,
                "model_id": model_id,
                "evaluation_iteration": iteration,
                "n_observations": len(group),
                "n_groups": group["evaluation_group_id"].nunique(),
                **values,
            }
        )
    for (scheme_id, model_id), group in predictions.groupby(
        ["scheme_id", "model_id"], sort=True, dropna=False
    ):
        values = stage32._metric_values(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "scheme_id": scheme_id,
                "model_id": model_id,
                "evaluation_iteration": "pooled",
                "n_observations": len(group),
                "n_groups": group["evaluation_group_id"].nunique(),
                **values,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["scheme_id", "model_id", "evaluation_iteration"], kind="stable")
        .reset_index(drop=True)
    )


def _bootstrap_seed(scheme_id: str, model_b: str, metric: str) -> int:
    text = f"{BOOTSTRAP_SEED}|{scheme_id}|{MODEL_ID}|{model_b}|{metric}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % (
        2**32
    )


def _paired_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scheme_id in CORE_SCHEME_IDS:
        scheme = predictions.loc[predictions["scheme_id"].astype(str).eq(scheme_id)]
        left = scheme.loc[scheme["model_id"].eq(MODEL_ID)].copy()
        for reference_model_id in REFERENCE_MODEL_IDS:
            right = scheme.loc[scheme["model_id"].eq(reference_model_id)].copy()
            paired = left[
                [*COMPARISON_KEY, "evaluation_group_id", "y_true", "y_pred"]
            ].merge(
                right[[*COMPARISON_KEY, "y_true", "y_pred"]],
                on=list(COMPARISON_KEY),
                how="inner",
                validate="one_to_one",
                suffixes=("_a", "_b"),
            )
            if len(paired) != len(left) or len(paired) != len(right):
                raise AtomicLocalVoxelEvaluationError(
                    f"Paired population mismatch for {scheme_id}/{reference_model_id}"
                )
            if not np.allclose(
                paired["y_true_a"].to_numpy(dtype=float),
                paired["y_true_b"].to_numpy(dtype=float),
                rtol=0.0,
                atol=stage32.TARGET_ABSOLUTE_TOLERANCE,
            ):
                raise AtomicLocalVoxelEvaluationError(
                    f"Paired outcomes disagree for {scheme_id}/{reference_model_id}"
                )
            y_true = paired["y_true_a"].to_numpy(dtype=float)
            pred_a = paired["y_pred_a"].to_numpy(dtype=float)
            pred_b = paired["y_pred_b"].to_numpy(dtype=float)
            metrics_a = stage32._metric_values(y_true, pred_a)
            metrics_b = stage32._metric_values(y_true, pred_b)
            group_ids = sorted(paired["evaluation_group_id"].astype(str).unique())
            groups = {
                group_id: paired.loc[
                    paired["evaluation_group_id"].astype(str).eq(group_id)
                ]
                for group_id in group_ids
            }
            for metric in ("spearman", "pearson", "mae", "rmse"):
                observed = stage32._metric_difference(metric, metrics_a, metrics_b)
                rng = np.random.default_rng(
                    _bootstrap_seed(scheme_id, reference_model_id, metric)
                )
                bootstrap_values: list[float] = []
                for _ in range(BOOTSTRAP_REPLICATES):
                    sampled_ids = rng.choice(
                        group_ids, size=len(group_ids), replace=True
                    )
                    sampled = pd.concat(
                        [groups[group_id] for group_id in sampled_ids],
                        ignore_index=True,
                    )
                    truth = sampled["y_true_a"].to_numpy(dtype=float)
                    values_a = stage32._metric_values(
                        truth, sampled["y_pred_a"].to_numpy(dtype=float)
                    )
                    values_b = stage32._metric_values(
                        truth, sampled["y_pred_b"].to_numpy(dtype=float)
                    )
                    value = stage32._metric_difference(metric, values_a, values_b)
                    if math.isfinite(value):
                        bootstrap_values.append(value)
                if not bootstrap_values:
                    raise AtomicLocalVoxelEvaluationError(
                        f"No valid bootstrap replicates for {scheme_id}/{metric}"
                    )
                lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_a": MODEL_ID,
                        "model_b": reference_model_id,
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


def inspect_atomic_local_voxel_inputs(repo_root: Path) -> Mapping[str, object]:
    tables = stage32._load_tables(repo_root)
    stage32._validate_input_contract(tables)
    observations = stage32._build_analysis_observations(tables)
    contexts = stage32._make_contexts(tables, observations)
    voxel_bank, lookup, audit = _load_atomic_context(repo_root)
    if len(contexts) != EXPECTED_CONTEXT_COUNT:
        raise AtomicLocalVoxelEvaluationError(
            f"Unexpected evaluation context count: {len(contexts)}"
        )
    total_test_rows = 0
    for context in contexts:
        test_rows = _test_rows(context)
        _voxel_indices(test_rows, lookup)
        total_test_rows += len(test_rows)
    if total_test_rows != EXPECTED_PREDICTIONS_PER_MODEL:
        raise AtomicLocalVoxelEvaluationError(
            f"Unexpected evaluation-row total: {total_test_rows}"
        )
    comparator = _load_comparator_predictions(repo_root)
    return {
        "status": "PASS",
        "analysis_scope": (
            "atom_level_local_experimental_structure_mutation_effect_evaluation"
        ),
        "analysis_goal": (
            "whether atom-level local WT structural information contains "
            "mutation-effect information not captured by residue-level "
            "structural representations"
        ),
        "model_id": MODEL_ID,
        "evaluation_context_count": len(contexts),
        "evaluation_prediction_count": total_test_rows,
        "voxel_bank_shape": list(voxel_bank.shape),
        "voxel_grid_count": int(voxel_bank.shape[0]),
        "voxel_channels": list(VOXEL_CHANNELS),
        "voxel_size_angstrom": VOXEL_SIZE_ANGSTROM,
        "voxel_half_width_angstrom": VOXEL_HALF_WIDTH_ANGSTROM,
        "coordinate_frame": "backbone_N_CA_C_right_handed",
        "protein_atoms_only": True,
        "hetero_atoms_included": False,
        "metal_atoms_included": False,
        "predicted_mutant_structures_used": False,
        "target_outcome_used_for_voxel_construction": False,
        "model_selection_uses_test_labels": False,
        "comparator_models": list(REFERENCE_MODEL_IDS),
        "comparator_prediction_rows": len(comparator),
        "voxel_audit_rows": len(audit),
        "protein_grid_counts": audit["protein_id"]
        .value_counts()
        .sort_index()
        .to_dict(),
    }


def run_atomic_local_voxel_evaluation(
    repo_root: Path,
    device_name: str = "cuda",
    repeats: int = TRAINING_REPEATS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    postprocess_only: bool = False,
) -> Mapping[str, object]:
    _configure_reproducible_execution()
    if repeats < 1 or repeats > len(FINAL_REPEAT_SEEDS):
        raise AtomicLocalVoxelEvaluationError(
            f"repeats must be between 1 and {len(FINAL_REPEAT_SEEDS)}"
        )
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise AtomicLocalVoxelEvaluationError(
            "max_epochs, patience, and batch_size must be positive"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise AtomicLocalVoxelEvaluationError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    input_review = inspect_atomic_local_voxel_inputs(repo_root)
    tables = stage32._load_tables(repo_root)
    observations = stage32._build_analysis_observations(tables)
    contexts = stage32._make_contexts(tables, observations)
    voxel_bank_cpu, lookup, audit = _load_atomic_context(repo_root)
    voxel_bank = voxel_bank_cpu.to(device=device, dtype=torch.float32)

    output_dir = repo_root / "data/interim/atomic_local_voxel_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(audit, output_dir / "atomic_local_voxel_grid_audit.csv")
    contract = _checkpoint_contract(
        device, repeats, max_epochs, patience, batch_size, repo_root
    )
    checkpoint_dir = _prepare_checkpoint_directory(
        output_dir,
        contract,
        postprocess_only=postprocess_only,
    )
    run_record = json.loads(
        (checkpoint_dir / CHECKPOINT_RECORD_NAME).read_text(encoding="utf-8")
    )
    training_source_sha256 = str(run_record["source_sha256"])
    postprocessing_source_sha256 = _sha256(Path(__file__))

    repeat_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    overall = tqdm(
        total=len(contexts),
        desc="atomic local voxel analyses",
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
            raise AtomicLocalVoxelEvaluationError(
                "Post-processing requested but one or more context "
                "checkpoints are absent"
            )
        selected_epoch, selection_history = _selection_for_context(
            context, voxel_bank, lookup, device, max_epochs, patience, batch_size
        )
        fitting_rows = _fitting_rows(context)
        reference = context["reference"].copy().reset_index(drop=True)
        ordered_test = reference[["observation_id"]].merge(
            test_rows, on="observation_id", how="left", validate="one_to_one"
        )
        if ordered_test[TARGET_COLUMN].isna().any():
            raise AtomicLocalVoxelEvaluationError("Unable to order voxel-CNN test rows")
        repeat_rows: list[Mapping[str, object]] = []
        final_history: list[Mapping[str, object]] = []
        condition_mode = stage32.CONDITION_MODE_BY_SCHEME[str(context["scheme_id"])]
        for repeat_index, seed in enumerate(FINAL_REPEAT_SEEDS[:repeats], start=1):
            prediction, history = _fit_fixed_epochs_and_predict(
                fitting_rows,
                ordered_test,
                voxel_bank,
                lookup,
                condition_mode,
                device,
                seed,
                selected_epoch,
                f"{context['scheme_id']} "
                f"{context['evaluation_iteration']} {MODEL_ID} "
                f"repeat {repeat_index}/{repeats}",
                batch_size,
            )
            repeat_rows.extend(
                _prediction_records(context, ordered_test, prediction, repeat_index)
            )
            for row in history:
                final_history.append(
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
        repeat_frame = pd.DataFrame(repeat_rows)
        history_frame = pd.DataFrame([*selection_history, *final_history])
        _write_context_checkpoint(
            checkpoint_dir,
            context,
            repeat_frame,
            history_frame,
            selected_epoch,
            ordered_test,
            repeats,
        )
        checkpoint = _load_context_checkpoint(
            checkpoint_dir, context, ordered_test, repeats
        )
        if checkpoint is None:
            raise AtomicLocalVoxelEvaluationError(
                "Context checkpoint could not be reloaded after writing"
            )
        repeat_frames.append(checkpoint[0])
        history_frames.append(checkpoint[1])
        overall.update(1)
        overall.set_postfix(
            scheme=str(context["scheme_id"]),
            iteration=str(context["evaluation_iteration"]),
            selected_epochs=selected_epoch,
            status="trained",
            refresh=False,
        )
    overall.close()

    repeat_predictions = pd.concat(repeat_frames, ignore_index=True)
    averaged = _average_repeat_predictions(repeat_predictions)
    if len(averaged) != EXPECTED_PREDICTIONS_PER_MODEL:
        raise AtomicLocalVoxelEvaluationError(
            f"Unexpected averaged prediction count: {len(averaged)}"
        )
    comparator = _load_comparator_predictions(repo_root)
    all_predictions = pd.concat([averaged, comparator], ignore_index=True)
    metrics = _metrics(all_predictions)
    comparisons = _paired_comparisons(all_predictions)
    history = pd.concat(history_frames, ignore_index=True)

    _write_csv(averaged, output_dir / "atomic_local_voxel_predictions.csv")
    _write_csv(
        repeat_predictions, output_dir / "atomic_local_voxel_repeat_predictions.csv"
    )
    _write_csv(history, output_dir / "atomic_local_voxel_training_history.csv")
    _write_csv(metrics, output_dir / "atomic_local_voxel_metrics.csv")
    _write_csv(comparisons, output_dir / "atomic_local_voxel_paired_comparisons.csv")
    _write_json(
        {
            "status": "TRAINING_COMPLETE",
            "context_count": len(contexts),
            "prediction_count": len(averaged),
            "training_source_sha256": training_source_sha256,
            "postprocessing_source_sha256": postprocessing_source_sha256,
        },
        output_dir / TRAINING_COMPLETE_RECORD_NAME,
    )

    summary = {
        **input_review,
        "status": "PASS",
        "training_source_sha256": training_source_sha256,
        "postprocessing_source_sha256": postprocessing_source_sha256,
        "training_repeats": repeats,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "metrics_row_count": len(metrics),
        "paired_comparison_row_count": len(comparisons),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_resampling_unit": "evaluation_group_id",
        "test_labels_used_for_epoch_selection": False,
        "cross_protein_target_labels_used_for_epoch_selection": False,
        "output_directory": str(output_dir.relative_to(repo_root)),
    }
    _write_json(summary, output_dir / "atomic_local_voxel_summary.json")
    return summary
