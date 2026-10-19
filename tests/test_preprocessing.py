"""Unit tests for data cleaning and the preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURES,
    TARGET,
    build_preprocessor,
    clean_data,
    split_features_target,
)


def test_raw_data_has_expected_shape(raw_df):
    assert raw_df.shape == (303, 14)


def test_clean_data_creates_binary_target(clean_df):
    assert set(clean_df[TARGET].unique()) == {0, 1}
    assert "num" not in clean_df.columns


def test_target_mapping_is_correct():
    """num 0 -> 0 (no disease); num 1-4 -> 1 (disease)."""
    rows = [{**{f: 1 for f in FEATURES}, "num": n} for n in [0, 1, 2, 3, 4]]
    out = clean_data(pd.DataFrame(rows))
    assert out[TARGET].tolist() == [0, 0, 1, 1, 1]


def test_clean_data_removes_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.head(5)])
    assert len(clean_data(doubled)) == len(clean_data(raw_df))


def test_clean_data_converts_invalid_values_to_nan(raw_df):
    df = raw_df.head(3).copy()
    df["ca"] = df["ca"].astype(object)
    df.loc[0, "ca"] = "?"
    out = clean_data(df)
    assert np.isnan(out.loc[0, "ca"])


def test_clean_data_raises_on_missing_column(raw_df):
    with pytest.raises(ValueError, match="Missing columns"):
        clean_data(raw_df.drop(columns=["chol"]))


def test_clean_data_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_preprocessor_output_shape_and_no_nans(clean_df):
    X, _ = split_features_target(clean_df)
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape == (len(X), 22)
    assert not np.isnan(Xt).any(), "Imputation should remove all missing values"


def test_numeric_features_are_scaled(clean_df):
    X, _ = split_features_target(clean_df)
    Xt = build_preprocessor().fit_transform(X)
    numeric = Xt[:, :5]  # first 5 columns are the scaled numeric features
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-6)


def test_preprocessor_handles_unseen_category(clean_df):
    """A category never seen in training must not crash the API."""
    X, _ = split_features_target(clean_df)
    pre = build_preprocessor().fit(X)
    new = X.head(1).copy()
    new["cp"] = 99.0
    assert pre.transform(new).shape == (1, 22)
