# Prepare residue graphs from the selected experimental structures.

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class ResidueGraphError(RuntimeError):
    """Raised when experimental-structure residue graphs cannot be prepared safely."""


PRIMARY_CA_DISTANCE_CUTOFF_ANGSTROM = 8.0

CORE_SCHEME_IDS = (
    "ndm1_position_nested5",
    "vim2_2020_position_nested5",
    "ndm1_to_vim2_amp_source_cv5",
    "vim2_to_ndm1_amp_source_cv5",
    "vim2_2022_mutation_nested5",
)

EXPECTED_CURRENT_NODE_COUNTS = {
    "NDM-1": 238,
    "VIM-2": 231,
}

EXPECTED_CURRENT_EDGE_COUNTS = {
    "NDM-1": 1204,
    "VIM-2": 1212,
}

EXPECTED_CURRENT_TESTED_POSITION_COUNTS = {
    "NDM-1": 238,
    "VIM-2": 230,
}

EXPECTED_CURRENT_CONTEXT_ONLY_ASSAY_POSITIONS = {
    "NDM-1": [],
    "VIM-2": [186],
}

NODE_SOURCE_COLUMNS = (
    "protein_id",
    "assay_position",
    "assay_residue",
    "selected_pdb_id",
    "selected_pdb_chain",
    "pdb_coordinate_present",
    "ca_x",
    "ca_y",
    "ca_z",
    "side_chain_centroid_x",
    "side_chain_centroid_y",
    "side_chain_centroid_z",
    "heavy_atom_count",
    "sasa_angstrom2",
    "relative_sasa_wilke",
    "secondary_structure_3state",
)

NODE_OUTPUT_COLUMNS = (
    "protein_id",
    "node_index",
    "assay_position",
    "assay_residue",
    "pdb_id",
    "pdb_chain",
    "ca_x",
    "ca_y",
    "ca_z",
    "side_chain_centroid_x",
    "side_chain_centroid_y",
    "side_chain_centroid_z",
    "heavy_atom_count",
    "sasa_angstrom2",
    "relative_sasa_wilke",
    "secondary_structure_3state",
)

EDGE_SOURCE_COLUMNS = (
    "protein_id",
    "pdb_id",
    "pdb_chain",
    "left_assay_position",
    "right_assay_position",
    "sequence_separation",
    "peptide_adjacent",
    "ca_distance_angstrom",
    "minimum_heavy_atom_distance_angstrom",
    "heavy_atom_contact_4_5_angstrom",
)

EDGE_OUTPUT_COLUMNS = (
    "protein_id",
    "source_node_index",
    "target_node_index",
    "source_assay_position",
    "target_assay_position",
    "pdb_id",
    "pdb_chain",
    "ca_distance_angstrom",
    "minimum_heavy_atom_distance_angstrom",
    "sequence_separation",
    "peptide_adjacent",
    "heavy_atom_contact_4_5_angstrom",
)

RESIDUE_CONSTRUCTION_COLUMNS = {
    "protein_id",
    "assay_position",
    "assay_residue",
    "ca_x",
    "ca_y",
    "ca_z",
    "side_chain_centroid_x",
    "side_chain_centroid_y",
    "side_chain_centroid_z",
    "heavy_atom_count",
    "sasa_angstrom2",
    "relative_sasa_wilke",
    "secondary_structure_3state",
}

PAIR_CONSTRUCTION_COLUMNS = {
    "protein_id",
    "pdb_id",
    "pdb_chain",
    "left_assay_position",
    "right_assay_position",
    "sequence_separation",
    "peptide_adjacent",
    "ca_distance_angstrom",
    "minimum_heavy_atom_distance_angstrom",
    "heavy_atom_contact_4_5_angstrom",
}

FORBIDDEN_GRAPH_FIELDS = {
    "score_original",
    "score_analysis",
    "score_uncertainty",
    "y_true",
    "y_pred",
    "evaluation_iteration",
    "evaluation_role",
    "evaluation_group_id",
    "group_id",
    "split_role",
    "cv_fold",
    "selected_alpha",
    "selected_radius",
    "selected_hidden_state_index",
    "mcsa_reference_correspondence_count",
    "mcsa_reference_ids",
    "mcsa_reference_positions",
    "mcsa_correspondence_statuses",
    "mcsa_all_correspondences_identity_conserved",
    "mcsa_any_metal_ligand_role_correspondence",
    "mcsa_any_selected_catalytic_zinc_contact",
    "ca_distance_to_catalytic_zinc_1_angstrom",
    "ca_distance_to_catalytic_zinc_2_angstrom",
    "minimum_atom_distance_to_catalytic_zinc_1_angstrom",
    "minimum_atom_distance_to_catalytic_zinc_2_angstrom",
    "ca_distance_to_catalytic_zinc_centroid_angstrom",
    "ca_distance_to_nearest_mcsa_reference_correspondence_angstrom",
    "minimum_atom_distance_to_nearest_mcsa_reference_correspondence_angstrom",
    "ca_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
    "minimum_atom_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
    "reported_homologue_pair_position",
    "reported_homologue_partner_assay_position",
    "reported_homologue_pair_source",
    "reported_homologue_pair_scope",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ResidueGraphError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def _write_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    if frame.empty:
        raise ResidueGraphError(f"Refusing to write an empty table: {path}")
    if list(frame.columns) != list(columns):
        raise ResidueGraphError(
            f"Unexpected output columns for {path.name}: {list(frame.columns)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "", "nan", "none"}:
        return False
    raise ResidueGraphError(f"Cannot interpret boolean value: {value}")


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    table_name: str,
) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ResidueGraphError(
            f"{table_name} is missing required columns: {sorted(missing)}"
        )


def _validate_contracts(
    residue_contracts: pd.DataFrame,
    pair_contracts: pd.DataFrame,
) -> None:
    _require_columns(
        residue_contracts,
        (
            "column_name",
            "representation_construction_allowed",
            "biological_validation_reference",
        ),
        table_name="residue feature contracts",
    )
    _require_columns(
        pair_contracts,
        (
            "column_name",
            "graph_construction_allowed",
            "direct_predictor_allowed",
        ),
        table_name="residue-pair feature contracts",
    )

    residue_contract = residue_contracts.set_index("column_name")
    for column in sorted(RESIDUE_CONSTRUCTION_COLUMNS):
        if column not in residue_contract.index:
            raise ResidueGraphError(
                f"Residue graph source column is absent from the contract: {column}"
            )
        allowed = _as_bool(
            residue_contract.loc[column, "representation_construction_allowed"]
        )
        if not allowed:
            raise ResidueGraphError(
                f"Residue graph construction is not permitted to use {column}"
            )
        biological_reference = _as_bool(
            residue_contract.loc[column, "biological_validation_reference"]
        )
        if biological_reference:
            raise ResidueGraphError(
                f"Biological reference field entered graph construction: {column}"
            )

    pair_contract = pair_contracts.set_index("column_name")
    for column in sorted(PAIR_CONSTRUCTION_COLUMNS):
        if column not in pair_contract.index:
            raise ResidueGraphError(
                "Residue-pair graph source column is absent from the contract: "
                f"{column}"
            )
        allowed = _as_bool(pair_contract.loc[column, "graph_construction_allowed"])
        if not allowed:
            raise ResidueGraphError(
                f"Residue-pair graph construction is not permitted to use {column}"
            )


def _build_nodes(residues: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        residues,
        NODE_SOURCE_COLUMNS,
        table_name="residue features",
    )

    coordinate_mask = residues["pdb_coordinate_present"].map(_as_bool)
    nodes = residues.loc[coordinate_mask, list(NODE_SOURCE_COLUMNS)].copy()

    if nodes.empty:
        raise ResidueGraphError("No coordinate-supported residues are available")

    if nodes.duplicated(["protein_id", "assay_position"]).any():
        raise ResidueGraphError("Coordinate-supported residue keys are not unique")

    numeric_columns = (
        "assay_position",
        "ca_x",
        "ca_y",
        "ca_z",
        "heavy_atom_count",
        "sasa_angstrom2",
        "relative_sasa_wilke",
    )
    for column in numeric_columns:
        values = pd.to_numeric(nodes[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ResidueGraphError(
                f"Coordinate-supported node field is missing or non-finite: {column}"
            )

    if nodes["secondary_structure_3state"].isna().any():
        raise ResidueGraphError(
            "Coordinate-supported nodes contain missing secondary-structure assignments"
        )

    nodes = nodes.sort_values(
        ["protein_id", "assay_position"],
        kind="stable",
    ).reset_index(drop=True)

    nodes["node_index"] = nodes.groupby("protein_id", sort=False).cumcount()

    nodes = nodes.rename(
        columns={
            "selected_pdb_id": "pdb_id",
            "selected_pdb_chain": "pdb_chain",
        }
    )

    nodes = nodes[list(NODE_OUTPUT_COLUMNS)]

    for protein_id, group in nodes.groupby("protein_id", sort=True):
        expected = np.arange(len(group), dtype=int)
        observed = group["node_index"].to_numpy(dtype=int)
        if not np.array_equal(observed, expected):
            raise ResidueGraphError(f"Node indices are not contiguous for {protein_id}")

    return nodes


def _build_edges(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    ca_distance_cutoff_angstrom: float = PRIMARY_CA_DISTANCE_CUTOFF_ANGSTROM,
) -> pd.DataFrame:
    _require_columns(
        pairs,
        EDGE_SOURCE_COLUMNS,
        table_name="residue-pair geometry",
    )

    if not np.isfinite(ca_distance_cutoff_angstrom):
        raise ResidueGraphError("Graph distance cutoff must be finite")
    if ca_distance_cutoff_angstrom <= 0:
        raise ResidueGraphError("Graph distance cutoff must be positive")

    pair_distance = pd.to_numeric(
        pairs["ca_distance_angstrom"],
        errors="coerce",
    )
    if pair_distance.isna().any():
        raise ResidueGraphError("Residue-pair C-alpha distance contains missing values")
    if not np.isfinite(pair_distance.to_numpy(dtype=float)).all():
        raise ResidueGraphError(
            "Residue-pair C-alpha distance contains non-finite values"
        )
    if (pair_distance <= 0).any():
        raise ResidueGraphError("Residue-pair C-alpha distance must be positive")

    selected = pairs.loc[
        pair_distance <= ca_distance_cutoff_angstrom,
        list(EDGE_SOURCE_COLUMNS),
    ].copy()

    if selected.empty:
        raise ResidueGraphError("Primary graph contains no edges")

    left_nodes = nodes[["protein_id", "assay_position", "node_index"]].rename(
        columns={
            "assay_position": "left_assay_position",
            "node_index": "source_node_index",
        }
    )
    right_nodes = nodes[["protein_id", "assay_position", "node_index"]].rename(
        columns={
            "assay_position": "right_assay_position",
            "node_index": "target_node_index",
        }
    )

    selected = selected.merge(
        left_nodes,
        on=["protein_id", "left_assay_position"],
        how="left",
        validate="many_to_one",
    )
    selected = selected.merge(
        right_nodes,
        on=["protein_id", "right_assay_position"],
        how="left",
        validate="many_to_one",
    )

    if selected["source_node_index"].isna().any():
        raise ResidueGraphError("Primary edge has a missing source node")
    if selected["target_node_index"].isna().any():
        raise ResidueGraphError("Primary edge has a missing target node")

    selected["source_node_index"] = selected["source_node_index"].astype(int)
    selected["target_node_index"] = selected["target_node_index"].astype(int)

    if (selected["left_assay_position"] >= selected["right_assay_position"]).any():
        raise ResidueGraphError(
            "Residue-pair table must store the lower assay position first"
        )

    if (selected["source_node_index"] == selected["target_node_index"]).any():
        raise ResidueGraphError("Primary graph contains a self-loop")

    if selected.duplicated(
        ["protein_id", "source_node_index", "target_node_index"]
    ).any():
        raise ResidueGraphError("Primary graph contains duplicate unordered edges")

    edges = selected.rename(
        columns={
            "left_assay_position": "source_assay_position",
            "right_assay_position": "target_assay_position",
        }
    )

    edges = (
        edges[list(EDGE_OUTPUT_COLUMNS)]
        .sort_values(
            ["protein_id", "source_node_index", "target_node_index"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    forbidden = FORBIDDEN_GRAPH_FIELDS.intersection(edges.columns)
    if forbidden:
        raise ResidueGraphError(
            f"Forbidden fields entered graph edges: {sorted(forbidden)}"
        )

    return edges


def _component_sizes(node_count: int, edges: pd.DataFrame) -> list[int]:
    if node_count <= 0:
        raise ResidueGraphError("Graph must contain at least one node")

    adjacency = [set() for _ in range(node_count)]

    for row in edges.itertuples(index=False):
        left = int(row.source_node_index)
        right = int(row.target_node_index)
        if left < 0 or left >= node_count or right < 0 or right >= node_count:
            raise ResidueGraphError("Graph edge references an invalid node index")
        adjacency[left].add(right)
        adjacency[right].add(left)

    seen: set[int] = set()
    sizes: list[int] = []

    for start in range(node_count):
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        size = 0

        while stack:
            current = stack.pop()
            size += 1
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)

        sizes.append(size)

    return sorted(sizes, reverse=True)


def _graph_statistics(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict[str, Mapping[str, object]]:
    statistics: dict[str, Mapping[str, object]] = {}

    for protein_id, node_group in nodes.groupby("protein_id", sort=True):
        edge_group = edges[edges["protein_id"].eq(protein_id)].copy()
        node_count = len(node_group)
        edge_count = len(edge_group)

        degrees = np.zeros(node_count, dtype=int)
        for row in edge_group.itertuples(index=False):
            degrees[int(row.source_node_index)] += 1
            degrees[int(row.target_node_index)] += 1

        component_sizes = _component_sizes(node_count, edge_group)

        statistics[str(protein_id)] = {
            "node_count": int(node_count),
            "edge_count": int(edge_count),
            "mean_degree": float(degrees.mean()),
            "median_degree": float(np.median(degrees)),
            "minimum_degree": int(degrees.min()),
            "maximum_degree": int(degrees.max()),
            "isolated_node_count": int((degrees == 0).sum()),
            "connected_component_count": int(len(component_sizes)),
            "largest_component_node_count": int(component_sizes[0]),
        }

    return statistics


def _tested_positions(membership: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        membership,
        (
            "scheme_id",
            "protein_id",
            "assay_position",
            "experimental_structure_task_eligible",
        ),
        table_name="modelling membership",
    )

    selected = membership[
        membership["scheme_id"].isin(CORE_SCHEME_IDS)
        & membership["experimental_structure_task_eligible"].map(_as_bool)
    ][["protein_id", "assay_position"]].copy()

    if selected.empty:
        raise ResidueGraphError(
            "No experimental-structure-eligible core evaluation positions are available"
        )

    return (
        selected.drop_duplicates()
        .sort_values(["protein_id", "assay_position"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    tested_positions: pd.DataFrame,
    selected_structures: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, list[int]]]:
    forbidden_nodes = FORBIDDEN_GRAPH_FIELDS.intersection(nodes.columns)
    forbidden_edges = FORBIDDEN_GRAPH_FIELDS.intersection(edges.columns)
    if forbidden_nodes or forbidden_edges:
        raise ResidueGraphError(
            "Forbidden outcome/evaluation/reference fields entered graph outputs"
        )

    node_counts = {
        str(protein_id): int(len(group))
        for protein_id, group in nodes.groupby("protein_id", sort=True)
    }
    edge_counts = {
        str(protein_id): int(len(group))
        for protein_id, group in edges.groupby("protein_id", sort=True)
    }

    graph_keys = nodes[["protein_id", "assay_position"]].copy()
    coverage = tested_positions.merge(
        graph_keys.assign(_graph_node=True),
        on=["protein_id", "assay_position"],
        how="left",
        validate="one_to_one",
    )
    covered = coverage["_graph_node"].eq(True)
    if not covered.all():
        missing = coverage.loc[
            ~covered,
            ["protein_id", "assay_position"],
        ]
        raise ResidueGraphError(
            "Experimental-structure test positions are absent from the graph: "
            f"{missing.to_dict(orient='records')}"
        )

    tested_counts = {
        str(protein_id): int(len(group))
        for protein_id, group in tested_positions.groupby("protein_id", sort=True)
    }

    tested_key_set = set(
        zip(
            tested_positions["protein_id"].astype(str),
            tested_positions["assay_position"].astype(int),
        )
    )

    context_only: dict[str, list[int]] = {}
    for protein_id, group in nodes.groupby("protein_id", sort=True):
        positions = [
            int(position)
            for position in group["assay_position"]
            if (str(protein_id), int(position)) not in tested_key_set
        ]
        context_only[str(protein_id)] = positions

    graph_stats = _graph_statistics(nodes, edges)
    for protein_id, values in graph_stats.items():
        if int(values["isolated_node_count"]) != 0:
            raise ResidueGraphError(
                f"Primary graph contains isolated nodes for {protein_id}"
            )
        if int(values["connected_component_count"]) != 1:
            raise ResidueGraphError(f"Primary graph is disconnected for {protein_id}")

    for protein_id, structure in selected_structures.items():
        if protein_id not in node_counts:
            continue

        expected_pdb = str(structure["pdb_id"])
        expected_chain = str(structure["primary_chain"])
        observed = nodes[nodes["protein_id"].eq(protein_id)]

        if set(observed["pdb_id"].astype(str)) != {expected_pdb}:
            raise ResidueGraphError(
                f"Graph PDB identity disagrees with selected structure for {protein_id}"
            )
        if set(observed["pdb_chain"].astype(str)) != {expected_chain}:
            raise ResidueGraphError(
                "Graph chain identity disagrees with selected structure for "
                f"{protein_id}"
            )

    return node_counts, edge_counts, tested_counts, context_only


def prepare_residue_graphs(repo_root: Path) -> Mapping[str, object]:
    """Prepare outcome-blind residue graphs from selected experimental coordinates."""

    repo_root = repo_root.resolve()

    residue_path = repo_root / "data/processed/residue_features.csv"
    pair_path = repo_root / "data/processed/residue_pair_geometry.csv"
    membership_path = (
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv"
    )
    residue_contract_path = (
        repo_root / "data/interim/modelling_views/residue_feature_contracts.csv"
    )
    pair_contract_path = (
        repo_root / "data/interim/modelling_views/residue_pair_feature_contracts.csv"
    )
    selected_structure_path = (
        repo_root / "data/interim/protein_structures/selected_structures.json"
    )

    input_paths = {
        "residue_features": residue_path,
        "residue_pair_geometry": pair_path,
        "scheme_observation_membership": membership_path,
        "residue_feature_contracts": residue_contract_path,
        "residue_pair_feature_contracts": pair_contract_path,
        "selected_structures": selected_structure_path,
    }

    for name, path in input_paths.items():
        if not path.is_file():
            raise ResidueGraphError(f"Missing required {name} input: {path}")

    residues = _read_csv(residue_path)
    pairs = _read_csv(pair_path)
    membership = _read_csv(membership_path)
    residue_contracts = _read_csv(residue_contract_path)
    pair_contracts = _read_csv(pair_contract_path)

    selected_structures = json.loads(
        selected_structure_path.read_text(encoding="utf-8")
    )

    _validate_contracts(residue_contracts, pair_contracts)

    nodes = _build_nodes(residues)
    edges = _build_edges(pairs, nodes)
    tested_positions = _tested_positions(membership)

    node_counts, edge_counts, tested_counts, context_only = _validate_graph(
        nodes,
        edges,
        tested_positions,
        selected_structures,
    )

    if node_counts != EXPECTED_CURRENT_NODE_COUNTS:
        raise ResidueGraphError(f"Unexpected graph node counts: {node_counts}")
    if edge_counts != EXPECTED_CURRENT_EDGE_COUNTS:
        raise ResidueGraphError(f"Unexpected primary graph edge counts: {edge_counts}")
    if tested_counts != EXPECTED_CURRENT_TESTED_POSITION_COUNTS:
        raise ResidueGraphError(
            f"Unexpected graph-covered tested-position counts: {tested_counts}"
        )
    if context_only != EXPECTED_CURRENT_CONTEXT_ONLY_ASSAY_POSITIONS:
        raise ResidueGraphError(
            f"Unexpected context-only graph positions: {context_only}"
        )

    graph_statistics = _graph_statistics(nodes, edges)

    output_dir = repo_root / "data/interim/residue_graphs"
    node_output = output_dir / "residue_graph_nodes.csv"
    edge_output = output_dir / "residue_graph_edges.csv"
    summary_output = output_dir / "residue_graph_summary.json"

    _write_csv(nodes, node_output, NODE_OUTPUT_COLUMNS)
    _write_csv(edges, edge_output, EDGE_OUTPUT_COLUMNS)

    input_sha256 = {
        name: _sha256_file(path) for name, path in sorted(input_paths.items())
    }

    summary: Mapping[str, object] = {
        "status": "PASS",
        "analysis_scope": "experimental_structure_residue_graph_preparation",
        "coordinate_source": "selected_experimental_pdb",
        "primary_edge_definition": (
            f"undirected residue pair with C-alpha distance <= "
            f"{PRIMARY_CA_DISTANCE_CUTOFF_ANGSTROM:g} angstrom"
        ),
        "primary_ca_distance_cutoff_angstrom": PRIMARY_CA_DISTANCE_CUTOFF_ANGSTROM,
        "edge_directionality": "undirected_canonical_single_row",
        "self_loops_in_edge_table": False,
        "peptide_edges_added_separately": False,
        "peptide_adjacency_retained_as_edge_metadata": True,
        "continuous_ca_distance_retained": True,
        "minimum_heavy_atom_distance_retained": True,
        "node_index_scope": "zero_based_contiguous_within_protein",
        "node_order": "protein_id_then_assay_position",
        "edge_order": "protein_id_then_source_node_index_then_target_node_index",
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "tested_position_counts": tested_counts,
        "context_only_assay_positions": context_only,
        "graph_statistics": graph_statistics,
        "core_scheme_ids_used_for_coverage_audit": list(CORE_SCHEME_IDS),
        "all_core_tested_positions_have_graph_nodes": True,
        "outcome_values_used_for_graph_construction": False,
        "evaluation_assignments_used_for_graph_construction": False,
        "evaluation_membership_used_only_for_coverage_audit": True,
        "mechanistic_reference_fields_used_for_graph_construction": False,
        "reported_homologue_fields_used_for_graph_construction": False,
        "graph_model_fitted": False,
        "model_selection_performed": False,
        "input_sha256": input_sha256,
        "residue_graph_nodes_sha256": _sha256_file(node_output),
        "residue_graph_edges_sha256": _sha256_file(edge_output),
        "outputs": [
            "residue_graph_nodes.csv",
            "residue_graph_edges.csv",
            "residue_graph_summary.json",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return summary
