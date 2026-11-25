import numpy as np
import pandas as pd

from src.data_preprocessing import (
    ENGINEERED_COLS, NUM_COLS, TARGET, add_features, build_preprocessor,
    clean_data, split_data,
)


def test_add_features_values():
    df = pd.DataFrame({
        "total_rooms": [1000.0], "total_bedrooms": [200.0],
        "population": [600.0], "households": [250.0],
    })
    out = add_features(df)
    assert out.loc[0, "rooms_per_household"] == 4.0
    assert out.loc[0, "bedrooms_per_room"] == 0.2
    assert out.loc[0, "population_per_household"] == 2.4


def test_add_features_zero_households_gives_nan_not_inf():
    df = pd.DataFrame({
        "total_rooms": [100.0], "total_bedrooms": [20.0],
        "population": [50.0], "households": [0.0],
    })
    out = add_features(df)
    assert out["rooms_per_household"].isna().all()
    assert not np.isinf(out[ENGINEERED_COLS].to_numpy(dtype=float)).any()


def test_add_features_does_not_change_input(tiny_df):
    before = tiny_df.copy()
    add_features(tiny_df)
    pd.testing.assert_frame_equal(tiny_df, before)


def test_clean_data_drops_duplicates(tiny_df):
    with_dup = pd.concat([tiny_df, tiny_df.iloc[[0]]], ignore_index=True)
    assert len(clean_data(with_dup)) == len(tiny_df)


def test_split_sizes_and_no_overlap(tiny_df):
    X_train, X_test, y_train, y_test = split_data(tiny_df)
    assert len(X_train) == 48 and len(X_test) == 12
    assert TARGET not in X_train.columns
    assert set(X_train.index).isdisjoint(X_test.index)


def test_preprocessor_handles_missing_and_unknown_category(tiny_df):
    X = add_features(tiny_df).drop(columns=TARGET)
    X.loc[X.index[0], "total_bedrooms"] = np.nan
    prep = build_preprocessor().fit(X)
    new = X.iloc[[1]].copy()
    new["ocean_proximity"] = "SOMEWHERE NEW"
    assert not np.isnan(np.asarray(prep.transform(new))).any()


def test_num_cols_include_engineered_features():
    assert set(ENGINEERED_COLS) <= set(NUM_COLS)