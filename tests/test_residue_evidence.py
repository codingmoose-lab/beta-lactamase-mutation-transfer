from __future__ import annotations

import pandas as pd
import pytest

from betalactamase_transfer.residue_evidence import (
    ResidueEvidenceError,
    _require_unique_keys,
)


def test_require_unique_keys_rejects_ambiguous_rows() -> None:
    frame = pd.DataFrame(
        [
            {"protein_id": "VIM-2", "assay_position": 10},
            {"protein_id": "VIM-2", "assay_position": 10},
        ]
    )
    with pytest.raises(ResidueEvidenceError, match="duplicate key rows"):
        _require_unique_keys(frame, ["protein_id", "assay_position"], "test table")
