import io

import numpy as np
import pandas as pd

from src.data_preprocessing import (
    MIN_ROWS_ERROR, TARGET, read_csv_safely, validate_dataset,
)


def _big(tiny_df, n=120):
    return pd.concat([tiny_df] * 2, ignore_index=True).iloc[:n].assign(
        longitude=lambda d: d["longitude"] + np.arange(len(d)) * 1e-4   # keep rows unique
    )


def test_valid_dataset_passes(tiny_df):
    result = validate_dataset(_big(tiny_df))
    assert result.ok and result.errors == []
    assert len(result.data) == 120


def test_wrong_file_type():
    df, err = read_csv_safely(io.StringIO("a,b\n1,2"), name="data.xlsx")
    assert df is None and "csv" in err.lower()


def test_empty_file():
    df, err = read_csv_safely(io.StringIO(""), name="data.csv")
    assert df is None and "empty" in err.lower()


def test_header_only_csv_has_no_rows():
    df, err = read_csv_safely(io.StringIO("a,b\n"), name="data.csv")
    result = validate_dataset(df)
    assert not result.ok and "no data rows" in result.errors[0]


def test_missing_target_column(tiny_df):
    result = validate_dataset(_big(tiny_df).drop(columns=TARGET))
    assert not result.ok
    assert TARGET in result.errors[0]


def test_missing_feature_column_is_named(tiny_df):
    result = validate_dataset(_big(tiny_df).drop(columns="median_income"))
    assert not result.ok and any("median_income" in e for e in result.errors)


def test_non_numeric_target_rows_are_removed_with_warning(tiny_df):
    df = _big(tiny_df).astype({TARGET: object})
    df.loc[0, TARGET] = "unknown"
    result = validate_dataset(df)
    assert result.ok and len(result.data) == 119
    assert any("removed" in w for w in result.warnings)


def test_all_target_values_invalid(tiny_df):
    df = _big(tiny_df).astype({TARGET: object})
    df[TARGET] = "n/a"
    result = validate_dataset(df)
    assert not result.ok


def test_duplicates_are_removed_and_reported(tiny_df):
    df = pd.concat([_big(tiny_df), _big(tiny_df).iloc[[0]]], ignore_index=True)
    result = validate_dataset(df)
    assert result.ok and result.summary["duplicate_rows"] == 1


def test_missing_feature_values_are_reported_not_removed(tiny_df):
    df = _big(tiny_df)
    df.loc[0, "total_bedrooms"] = np.nan
    result = validate_dataset(df)
    assert result.ok and len(result.data) == 120
    assert result.summary["missing_values"]["total_bedrooms"] == 1


def test_too_small_dataset_is_rejected(tiny_df):
    result = validate_dataset(tiny_df.iloc[: MIN_ROWS_ERROR - 1])
    assert not result.ok


def test_small_dataset_gets_warning(tiny_df):
    result = validate_dataset(tiny_df)          # 60 rows
    assert result.ok and any("Small dataset" in w for w in result.warnings)