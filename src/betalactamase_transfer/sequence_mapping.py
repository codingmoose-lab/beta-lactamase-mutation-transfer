# Sequence and structure-reference auditing for the beta-lactamase study.
# ruff: noqa: UP045

from __future__ import annotations

import csv
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from Bio import BiopythonParserWarning, SeqIO
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1_extended
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
MISSING_CIF_VALUES = {"", ".", "?"}


class SequenceMappingError(RuntimeError):
    """Raised when an input cannot be audited without an unsafe assumption."""


@dataclass(frozen=True)
class GenBankTargetSpec:
    """Description of an assay construct feature in a GenBank file."""

    protein_id: str
    feature_types: Tuple[str, ...]
    required_label_terms: Tuple[str, ...]
    expected_protein_length: int
    record_id: str


@dataclass(frozen=True)
class SequenceRecord:
    """One sequence used in the reference audit."""

    record_id: str
    protein_id: str
    source_type: str
    source_file: str
    sequence: str
    description: str
    chain_ids: str = ""
    selection_status: str = ""
    feature_type: str = ""
    feature_label: str = ""
    feature_location: str = ""
    translation_frame: Optional[int] = None
    terminal_stop_found: Optional[bool] = None
    trailing_codons_after_stop: Optional[int] = None

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class PDBSchemeRow:
    """One deposited polymer position for one PDB chain."""

    pdb_id: str
    entity_id: str
    label_asym_id: str
    author_chain_id: str
    entity_position: int
    residue_three_letter: str
    residue_one_letter: str
    modeled: bool
    pdb_label_position: str
    pdb_author_position: str
    insertion_code: str


@dataclass(frozen=True)
class AlignmentConsensus:
    """Consensus correspondence across all equally scoring optimal alignments."""

    reference_to_query: Dict[int, Optional[int]]
    query_to_reference: Dict[int, Optional[int]]
    representative_alignment: object
    optimal_alignment_count: int
    ambiguous_reference_positions: Tuple[int, ...]
    ambiguous_query_positions: Tuple[int, ...]

    @property
    def alignment_score(self) -> float:
        return float(self.representative_alignment.score)


def clean_protein_sequence(value: str) -> str:
    """Normalize a protein sequence without silently deleting internal stops."""
    sequence = re.sub(r"\s+", "", value).upper()
    invalid = sorted(set(sequence) - VALID_AA)
    if invalid:
        raise SequenceMappingError(
            "Protein sequence contains unsupported characters: " + "".join(invalid)
        )
    if not sequence:
        raise SequenceMappingError("Protein sequence is empty after normalization")
    return sequence


def read_single_fasta(
    path: Path,
    protein_id: str,
    record_id: str,
) -> SequenceRecord:
    """Read exactly one non-empty FASTA record."""
    records = list(SeqIO.parse(str(path), "fasta"))
    if len(records) != 1:
        raise SequenceMappingError(
            f"Expected exactly one FASTA record in {path}, found {len(records)}"
        )
    record = records[0]
    sequence = clean_protein_sequence(str(record.seq))
    return SequenceRecord(
        record_id=record_id,
        protein_id=protein_id,
        source_type="uniprot_fasta",
        source_file=str(path),
        sequence=sequence,
        description=record.description,
        selection_status="selected_reference",
    )


def _qualifier_values(feature: object, key: str) -> List[str]:
    qualifiers = getattr(feature, "qualifiers", {})
    values = qualifiers.get(key, [])
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(values)]


def _feature_label_text(feature: object) -> str:
    values: List[str] = []
    for key in ("label", "gene", "product", "note", "locus_tag"):
        values.extend(_qualifier_values(feature, key))
    return " ".join(values).strip()


def _translate_nucleotide_feature(
    nucleotide_sequence: object,
    codon_start: int,
) -> Tuple[str, int, bool, int]:
    """Translate one feature and stop at the first in-frame stop codon.

    The first stop terminates the biological ORF. Any complete codons after that
    stop are reported rather than silently folded into the protein sequence.
    """
    if codon_start not in (1, 2, 3):
        raise SequenceMappingError("codon_start must be 1, 2, or 3")
    nucleotide = nucleotide_sequence[codon_start - 1 :]
    usable_length = len(nucleotide) - (len(nucleotide) % 3)
    if usable_length < 3:
        raise SequenceMappingError("Feature is too short to translate")
    translated = str(nucleotide[:usable_length].translate()).upper()
    if "*" in translated:
        stop_index = translated.index("*")
        protein = translated[:stop_index]
        trailing = len(translated) - stop_index - 1
        terminal_stop_found = True
    else:
        protein = translated
        trailing = 0
        terminal_stop_found = False
    if "*" in protein:
        raise SequenceMappingError(
            "Internal stop remained after translation truncation"
        )
    return clean_protein_sequence(protein), codon_start, terminal_stop_found, trailing


def extract_genbank_target(
    path: Path,
    spec: GenBankTargetSpec,
) -> SequenceRecord:
    """Extract a labelled assay ORF from a GenBank plasmid.

    No longest-feature fallback is used. A unique label/type match is required.
    Translation qualifiers are used when present; otherwise the nucleotide
    feature is translated with strand and compound-location handling delegated
    to Biopython's ``feature.extract``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonParserWarning)
        records = list(SeqIO.parse(str(path), "genbank"))
    if not records:
        raise SequenceMappingError(f"No GenBank records found in {path}")

    matches: List[Tuple[object, object, str]] = []
    for record in records:
        for feature in record.features:
            if feature.type not in spec.feature_types:
                continue
            label_text = _feature_label_text(feature)
            normalized = re.sub(r"[^a-z0-9]+", "", label_text.lower())
            if all(
                re.sub(r"[^a-z0-9]+", "", term.lower()) in normalized
                for term in spec.required_label_terms
            ):
                matches.append((record, feature, label_text))

    if len(matches) != 1:
        available = []
        for record in records:
            for feature in record.features:
                if feature.type in spec.feature_types:
                    available.append(
                        "{}:{}:{}".format(
                            feature.type,
                            feature.location,
                            _feature_label_text(feature) or "<unlabelled>",
                        )
                    )
        raise SequenceMappingError(
            "Expected one labelled target feature for {} in {}; "
            "found {}. Candidates: {}".format(
                spec.protein_id,
                path,
                len(matches),
                " | ".join(available) or "none",
            )
        )

    record, feature, label_text = matches[0]
    translations = _qualifier_values(feature, "translation")
    codon_start_values = _qualifier_values(feature, "codon_start") or ["1"]
    try:
        codon_start = int(codon_start_values[0])
    except ValueError as exc:
        raise SequenceMappingError(
            f"Invalid codon_start for {path}: {codon_start_values[0]}"
        ) from exc

    if translations:
        raw_translation = re.sub(r"\s+", "", translations[0]).upper()
        if "*" in raw_translation[:-1]:
            raise SequenceMappingError(
                f"Translation qualifier contains an internal stop for {path}"
            )
        terminal_stop = raw_translation.endswith("*")
        sequence = clean_protein_sequence(raw_translation.rstrip("*"))
        trailing_codons = 0
        translation_frame = codon_start
    else:
        nucleotide = feature.extract(record.seq)
        (
            sequence,
            translation_frame,
            terminal_stop,
            trailing_codons,
        ) = _translate_nucleotide_feature(nucleotide, codon_start)

    if len(sequence) != spec.expected_protein_length:
        raise SequenceMappingError(
            f"Extracted {spec.protein_id} sequence from {path} "
            f"has length {len(sequence)}, expected "
            f"{spec.expected_protein_length}. "
            f"Feature={feature.location}, label={label_text}, "
            f"frame={translation_frame}, stop={terminal_stop}, "
            f"trailing_codons={trailing_codons}"
        )

    description = (
        "GenBank record={}; feature_type={}; label={}; location={}; "
        "frame={}; terminal_stop={}; trailing_codons_after_stop={}"
    ).format(
        record.id,
        feature.type,
        label_text or "not supplied",
        feature.location,
        translation_frame,
        terminal_stop,
        trailing_codons,
    )
    return SequenceRecord(
        record_id=spec.record_id,
        protein_id=spec.protein_id,
        source_type="genbank_assay_construct",
        source_file=str(path),
        sequence=sequence,
        description=description,
        selection_status="genbank_assay_construct_feature_match",
        feature_type=feature.type,
        feature_label=label_text,
        feature_location=str(feature.location),
        translation_frame=translation_frame,
        terminal_stop_found=terminal_stop,
        trailing_codons_after_stop=trailing_codons,
    )


def _as_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def read_mmcif_entity_sequences(
    path: Path,
    protein_id: str,
    pdb_id: str,
) -> List[SequenceRecord]:
    """Read deposited polymer entity sequences from an mmCIF file."""
    data = MMCIF2Dict(str(path))
    entity_ids = _as_list(data.get("_entity_poly.entity_id"))
    sequences = _as_list(data.get("_entity_poly.pdbx_seq_one_letter_code_can"))
    strand_ids = _as_list(data.get("_entity_poly.pdbx_strand_id"))
    if not entity_ids or len(entity_ids) != len(sequences):
        raise SequenceMappingError(f"Invalid polymer entity sequence table in {path}")
    while len(strand_ids) < len(entity_ids):
        strand_ids.append("")
    output: List[SequenceRecord] = []
    for entity_id, sequence_value, chains in zip(entity_ids, sequences, strand_ids):
        output.append(
            SequenceRecord(
                record_id=f"{pdb_id}_entity_{entity_id}",
                protein_id=protein_id,
                source_type="pdb_deposited_polymer_sequence",
                source_file=str(path),
                sequence=clean_protein_sequence(sequence_value),
                description=f"PDB {pdb_id}; polymer entity {entity_id}",
                chain_ids=chains,
                selection_status="candidate_structure_sequence",
            )
        )
    return output


def _three_to_one(value: str) -> str:
    normalized = value.strip().title()
    return protein_letters_3to1_extended.get(normalized, "X")


def read_mmcif_scheme(path: Path, pdb_id: str) -> List[PDBSchemeRow]:
    """Read chain-specific deposited/modelled residue numbering from mmCIF."""
    data = MMCIF2Dict(str(path))
    columns = {
        "entity_id": _as_list(data.get("_pdbx_poly_seq_scheme.entity_id")),
        "asym_id": _as_list(data.get("_pdbx_poly_seq_scheme.asym_id")),
        "author_chain": _as_list(data.get("_pdbx_poly_seq_scheme.pdb_strand_id")),
        "seq_id": _as_list(data.get("_pdbx_poly_seq_scheme.seq_id")),
        "mon_id": _as_list(data.get("_pdbx_poly_seq_scheme.mon_id")),
        "pdb_mon_id": _as_list(data.get("_pdbx_poly_seq_scheme.pdb_mon_id")),
        "pdb_seq_num": _as_list(data.get("_pdbx_poly_seq_scheme.pdb_seq_num")),
        "auth_seq_num": _as_list(data.get("_pdbx_poly_seq_scheme.auth_seq_num")),
        "ins_code": _as_list(data.get("_pdbx_poly_seq_scheme.pdb_ins_code")),
    }
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise SequenceMappingError(f"Incomplete _pdbx_poly_seq_scheme table in {path}")
    rows: List[PDBSchemeRow] = []
    for index in range(next(iter(lengths))):
        pdb_mon_id = columns["pdb_mon_id"][index]
        auth_seq_num = columns["auth_seq_num"][index]
        modeled = (
            pdb_mon_id not in MISSING_CIF_VALUES
            and auth_seq_num not in MISSING_CIF_VALUES
        )
        rows.append(
            PDBSchemeRow(
                pdb_id=pdb_id,
                entity_id=columns["entity_id"][index],
                label_asym_id=columns["asym_id"][index],
                author_chain_id=columns["author_chain"][index],
                entity_position=int(columns["seq_id"][index]),
                residue_three_letter=columns["mon_id"][index],
                residue_one_letter=_three_to_one(columns["mon_id"][index]),
                modeled=modeled,
                pdb_label_position=columns["pdb_seq_num"][index],
                pdb_author_position=auth_seq_num,
                insertion_code=(
                    ""
                    if columns["ins_code"][index] in MISSING_CIF_VALUES
                    else columns["ins_code"][index]
                ),
            )
        )
    return rows


def _aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    return aligner


def _position_map_for_alignment(
    reference: str,
    query: str,
    alignment: object,
) -> Tuple[Dict[int, Optional[int]], Dict[int, Optional[int]]]:
    ref_to_query: Dict[int, Optional[int]] = {
        position: None for position in range(1, len(reference) + 1)
    }
    query_to_ref: Dict[int, Optional[int]] = {
        position: None for position in range(1, len(query) + 1)
    }
    coordinates = alignment.coordinates
    for block in range(coordinates.shape[1] - 1):
        ref_start = int(coordinates[0, block])
        ref_end = int(coordinates[0, block + 1])
        query_start = int(coordinates[1, block])
        query_end = int(coordinates[1, block + 1])
        ref_length = ref_end - ref_start
        query_length = query_end - query_start
        if ref_length == query_length and ref_length > 0:
            for offset in range(ref_length):
                ref_position = ref_start + offset + 1
                query_position = query_start + offset + 1
                ref_to_query[ref_position] = query_position
                query_to_ref[query_position] = ref_position
    return ref_to_query, query_to_ref


def alignment_consensus(
    reference: str,
    query: str,
) -> AlignmentConsensus:
    """Return correspondence shared by all equally scoring optimal alignments."""
    alignments = _aligner().align(reference, query)
    optimal_alignment_count = len(alignments)
    if optimal_alignment_count == 0:
        raise SequenceMappingError("PairwiseAligner returned no optimal alignment")

    representative = alignments[0]
    ref_maps: List[Dict[int, Optional[int]]] = []
    query_maps: List[Dict[int, Optional[int]]] = []
    for alignment in alignments:
        ref_map, query_map = _position_map_for_alignment(reference, query, alignment)
        ref_maps.append(ref_map)
        query_maps.append(query_map)

    ambiguous_reference = tuple(
        position
        for position in range(1, len(reference) + 1)
        if len({mapping[position] for mapping in ref_maps}) > 1
    )
    ambiguous_query = tuple(
        position
        for position in range(1, len(query) + 1)
        if len({mapping[position] for mapping in query_maps}) > 1
    )
    ref_to_query = {
        position: (
            ref_maps[0][position] if position not in ambiguous_reference else None
        )
        for position in range(1, len(reference) + 1)
    }
    query_to_ref = {
        position: (query_maps[0][position] if position not in ambiguous_query else None)
        for position in range(1, len(query) + 1)
    }
    return AlignmentConsensus(
        reference_to_query=ref_to_query,
        query_to_reference=query_to_ref,
        representative_alignment=representative,
        optimal_alignment_count=optimal_alignment_count,
        ambiguous_reference_positions=ambiguous_reference,
        ambiguous_query_positions=ambiguous_query,
    )


def alignment_position_map(
    reference: str,
    query: str,
) -> Tuple[Dict[int, Optional[int]], Dict[int, Optional[int]], object]:
    """Return one-based maps agreed on by every optimal global alignment."""
    consensus = alignment_consensus(reference, query)
    return (
        consensus.reference_to_query,
        consensus.query_to_reference,
        consensus.representative_alignment,
    )


def alignment_metrics(reference: str, query: str) -> Dict[str, object]:
    """Calculate identity, coverage, gap, and optimal-alignment diagnostics."""
    consensus = alignment_consensus(reference, query)
    ref_to_query, query_to_ref = _position_map_for_alignment(
        reference, query, consensus.representative_alignment
    )
    matches = 0
    substitutions = 0
    for ref_position, query_position in ref_to_query.items():
        if query_position is None:
            continue
        if reference[ref_position - 1] == query[query_position - 1]:
            matches += 1
        else:
            substitutions += 1
    aligned_overlap = matches + substitutions
    deletions = sum(value is None for value in ref_to_query.values())
    insertions = sum(value is None for value in query_to_ref.values())
    identity = matches / aligned_overlap if aligned_overlap else 0.0
    return {
        "alignment_score": round(consensus.alignment_score, 3),
        "optimal_alignment_count": consensus.optimal_alignment_count,
        "ambiguous_reference_position_count": len(
            consensus.ambiguous_reference_positions
        ),
        "ambiguous_reference_positions": ";".join(
            str(position) for position in consensus.ambiguous_reference_positions
        ),
        "ambiguous_query_position_count": len(consensus.ambiguous_query_positions),
        "ambiguous_query_positions": ";".join(
            str(position) for position in consensus.ambiguous_query_positions
        ),
        "matches": matches,
        "substitutions": substitutions,
        "insertions_relative_to_reference": insertions,
        "deletions_relative_to_reference": deletions,
        "aligned_overlap": aligned_overlap,
        "sequence_identity": round(identity, 6),
        "reference_coverage": round(aligned_overlap / len(reference), 6),
        "query_coverage": round(aligned_overlap / len(query), 6),
        "reference_length": len(reference),
        "query_length": len(query),
        "length_difference": len(query) - len(reference),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise SequenceMappingError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _scheme_index(
    rows: Sequence[PDBSchemeRow],
) -> Dict[Tuple[str, int], PDBSchemeRow]:
    return {(row.author_chain_id, row.entity_position): row for row in rows}


def _mapping_rows(
    protein_id: str,
    assay: SequenceRecord,
    uniprot: SequenceRecord,
    pdb_entity: SequenceRecord,
    scheme_rows: Sequence[PDBSchemeRow],
) -> List[Dict[str, object]]:
    """Build assay→UniProt→PDB chain mappings while retaining alignment gaps."""
    assay_consensus = alignment_consensus(uniprot.sequence, assay.sequence)
    pdb_consensus = alignment_consensus(uniprot.sequence, pdb_entity.sequence)
    if assay_consensus.ambiguous_query_positions:
        raise SequenceMappingError(
            f"{protein_id} assay-to-UniProt mapping is ambiguous at assay "
            f"positions {assay_consensus.ambiguous_query_positions}"
        )

    modeled_entity_positions = {
        row.entity_position for row in scheme_rows if row.modeled
    }
    ambiguous_modeled = sorted(
        modeled_entity_positions.intersection(pdb_consensus.ambiguous_query_positions)
    )
    if ambiguous_modeled:
        raise SequenceMappingError(
            f"{protein_id} UniProt-to-PDB alignment ambiguity affects modeled "
            f"PDB entity positions {ambiguous_modeled}"
        )

    assay_to_uniprot = assay_consensus.query_to_reference
    uniprot_to_entity = pdb_consensus.reference_to_query
    chains = sorted({row.author_chain_id for row in scheme_rows})
    scheme = _scheme_index(scheme_rows)
    rows: List[Dict[str, object]] = []
    for assay_position in range(1, len(assay.sequence) + 1):
        uniprot_position = assay_to_uniprot[assay_position]
        entity_position = (
            uniprot_to_entity.get(uniprot_position)
            if uniprot_position is not None
            else None
        )
        for chain in chains:
            scheme_row = (
                scheme.get((chain, entity_position))
                if entity_position is not None
                else None
            )
            assay_residue = assay.sequence[assay_position - 1]
            uniprot_residue = (
                uniprot.sequence[uniprot_position - 1]
                if uniprot_position is not None
                else ""
            )
            pdb_residue = (
                pdb_entity.sequence[entity_position - 1]
                if entity_position is not None
                else ""
            )
            rows.append(
                {
                    "protein_id": protein_id,
                    "assay_record_id": assay.record_id,
                    "assay_position": assay_position,
                    "assay_residue": assay_residue,
                    "uniprot_record_id": uniprot.record_id,
                    "uniprot_position": uniprot_position or "",
                    "uniprot_residue": uniprot_residue,
                    "pdb_id": scheme_rows[0].pdb_id,
                    "pdb_entity_id": pdb_entity.record_id,
                    "pdb_entity_position": entity_position or "",
                    "pdb_entity_residue": pdb_residue,
                    "pdb_chain": chain,
                    "pdb_label_asym_id": (
                        scheme_row.label_asym_id if scheme_row else ""
                    ),
                    "pdb_label_position": (
                        scheme_row.pdb_label_position if scheme_row else ""
                    ),
                    "pdb_author_position": (
                        scheme_row.pdb_author_position if scheme_row else ""
                    ),
                    "pdb_insertion_code": (
                        scheme_row.insertion_code if scheme_row else ""
                    ),
                    "pdb_modeled": bool(scheme_row and scheme_row.modeled),
                    "mapping_status": (
                        "assay_insertion"
                        if uniprot_position is None
                        else "pdb_alignment_ambiguous_unmodeled"
                        if uniprot_position
                        in pdb_consensus.ambiguous_reference_positions
                        else "not_in_pdb_entity"
                        if entity_position is None
                        else "mapped_modeled"
                        if scheme_row and scheme_row.modeled
                        else "mapped_unmodeled"
                    ),
                    "residue_identity_ok": bool(
                        uniprot_position is not None
                        and entity_position is not None
                        and assay_residue == uniprot_residue == pdb_residue
                    ),
                }
            )
    return rows


def build_sequence_mapping(repo_root: Path) -> Dict[str, object]:
    """Run the complete sequence audit and write interim outputs."""
    root = repo_root.resolve()
    paths = {
        "ndm_uniprot": root / "data/raw/uniprot/C7C422/C7C422.fasta",
        "vim_uniprot": root / "data/raw/uniprot/Q9K2N0/Q9K2N0.fasta",
        "ndm_genbank": root
        / "data/raw/ndm_vim_2024/chen_2024_supplementary_data_6_ndm1_plasmid.gb",
        "vim_genbank": root
        / "data/raw/vim2_2020/chen_2020_supplementary_file_1_vim2_plasmid.gb",
        "ndm_pdb": root / "data/raw/pdb/3SPU/3SPU.cif",
        "vim_pdb": root / "data/raw/pdb/5YD7/5YD7.cif",
        "a4grb6": root / "data/raw/uniprot/A4GRB6/A4GRB6.json",
    }
    missing = [
        str(path)
        for path in paths.values()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise SequenceMappingError(
            "Missing or empty required files: " + ", ".join(missing)
        )

    ndm_uniprot = read_single_fasta(
        paths["ndm_uniprot"], "NDM-1", "NDM1_UniProt_C7C422"
    )
    vim_uniprot = read_single_fasta(
        paths["vim_uniprot"], "VIM-2", "VIM2_UniProt_Q9K2N0"
    )
    ndm_assay = extract_genbank_target(
        paths["ndm_genbank"],
        GenBankTargetSpec(
            protein_id="NDM-1",
            feature_types=("exon", "CDS"),
            required_label_terms=("NDM1",),
            expected_protein_length=270,
            record_id="NDM1_2024_assay_construct",
        ),
    )
    vim_assay = extract_genbank_target(
        paths["vim_genbank"],
        GenBankTargetSpec(
            protein_id="VIM-2",
            feature_types=("CDS", "exon"),
            required_label_terms=("VIM", "2", "ORF"),
            expected_protein_length=267,
            record_id="VIM2_2020_assay_construct",
        ),
    )
    ndm_pdb_records = read_mmcif_entity_sequences(paths["ndm_pdb"], "NDM-1", "3SPU")
    vim_pdb_records = read_mmcif_entity_sequences(paths["vim_pdb"], "VIM-2", "5YD7")
    if len(ndm_pdb_records) != 1 or len(vim_pdb_records) != 1:
        raise SequenceMappingError(
            "Expected one protein polymer entity in each reference PDB"
        )
    ndm_pdb = ndm_pdb_records[0]
    vim_pdb = vim_pdb_records[0]
    sequences = [
        ndm_uniprot,
        ndm_assay,
        ndm_pdb,
        vim_uniprot,
        vim_assay,
        vim_pdb,
    ]

    output_dir = root / "data/interim/sequence_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: List[Dict[str, object]] = []
    for item in sequences:
        row = asdict(item)
        source_path = Path(item.source_file)
        try:
            row["source_file"] = source_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SequenceMappingError(
                f"Sequence source lies outside repository root: {source_path}"
            ) from exc
        row["length"] = item.length
        inventory_rows.append(row)
    _write_csv(output_dir / "sequence_sources.csv", inventory_rows)

    reference_by_protein = {
        "NDM-1": ndm_uniprot,
        "VIM-2": vim_uniprot,
    }
    comparison_rows: List[Dict[str, object]] = []
    for query in sequences:
        reference = reference_by_protein[query.protein_id]
        comparison_rows.append(
            {
                "protein_id": query.protein_id,
                "reference_record_id": reference.record_id,
                "query_record_id": query.record_id,
                "query_source_type": query.source_type,
                "query_chain_ids": query.chain_ids,
                "query_selection_status": query.selection_status,
                **alignment_metrics(reference.sequence, query.sequence),
            }
        )
    _write_csv(
        output_dir / "sequence_correspondence.csv",
        comparison_rows,
    )

    ndm_scheme = read_mmcif_scheme(paths["ndm_pdb"], "3SPU")
    vim_scheme = read_mmcif_scheme(paths["vim_pdb"], "5YD7")
    scheme_rows = [asdict(row) for row in ndm_scheme + vim_scheme]
    _write_csv(output_dir / "structure_residue_status.csv", scheme_rows)

    mapping_rows = _mapping_rows(
        "NDM-1", ndm_assay, ndm_uniprot, ndm_pdb, ndm_scheme
    ) + _mapping_rows("VIM-2", vim_assay, vim_uniprot, vim_pdb, vim_scheme)
    _write_csv(output_dir / "residue_number_mapping.csv", mapping_rows)

    with (output_dir / "reference_sequences.fasta").open(
        "w", encoding="utf-8"
    ) as handle:
        for item in sequences:
            handle.write(
                ">{} protein={} source={} chains={}\n".format(
                    item.record_id,
                    item.protein_id,
                    item.source_type,
                    item.chain_ids or "NA",
                )
            )
            for start in range(0, len(item.sequence), 80):
                handle.write(item.sequence[start : start + 80] + "\n")

    with paths["a4grb6"].open("r", encoding="utf-8") as handle:
        a4grb6 = json.load(handle)
    inactive_ok = (
        a4grb6.get("entryType") == "Inactive"
        and a4grb6.get("inactiveReason", {}).get("inactiveReasonType") == "DELETED"
    )
    if not inactive_ok:
        raise SequenceMappingError(
            "A4GRB6 status is not the expected inactive/deleted record"
        )

    summary = {
        "audit_status": "PASS",
        "assay_reference_decisions": {
            "NDM-1": ndm_assay.record_id,
            "VIM-2": vim_assay.record_id,
        },
        "database_reference_decisions": {
            "NDM-1": ndm_uniprot.record_id,
            "VIM-2": vim_uniprot.record_id,
            "A4GRB6": "historical_inactive_identifier_only",
        },
        "key_findings": {
            "ndm_assay_length": ndm_assay.length,
            "vim_assay_length": vim_assay.length,
            "ndm_assay_vs_uniprot": alignment_metrics(
                ndm_uniprot.sequence, ndm_assay.sequence
            ),
            "vim_assay_vs_uniprot": alignment_metrics(
                vim_uniprot.sequence, vim_assay.sequence
            ),
            "ndm_pdb_vs_uniprot": alignment_metrics(
                ndm_uniprot.sequence, ndm_pdb.sequence
            ),
            "vim_pdb_vs_uniprot": alignment_metrics(
                vim_uniprot.sequence, vim_pdb.sequence
            ),
        },
        "outputs": [
            "sequence_sources.csv",
            "sequence_correspondence.csv",
            "structure_residue_status.csv",
            "residue_number_mapping.csv",
            "reference_sequences.fasta",
        ],
    }
    with (output_dir / "sequence_mapping_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
