from __future__ import annotations

import pandas as pd

from betalactamase_transfer.processed_tables import _as_bool_series


def test_as_bool_series_handles_csv_style_values() -> None:
    series = pd.Series([True, False, "true", "false", 1, 0, None])
    result = _as_bool_series(series)
    assert result.tolist() == [True, False, True, False, True, False, False]
    assert str(result.dtype) == "bool"
