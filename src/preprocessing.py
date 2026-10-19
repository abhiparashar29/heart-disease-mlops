"""
Data cleaning and preprocessing for the heart disease project.

Shared by training (src/train.py), unit tests (tests/) and the API (app/),
so every stage transforms data in exactly the same way.

Usage:
    python src/preprocessing.py   -> writes data/processed/heart_clean.csv
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "heart_raw.csv"
CLEAN_FILE = ROOT / "data" / "processed" / "heart_clean.csv"

# Feature groups (decided during EDA)
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
NOMINAL_FEATURES = ["cp", "restecg", "slope", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
ORDINAL_FEATURES = ["ca"]
FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES

RAW_TARGET = "num"
TARGET = "target"


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the raw CSV. '?' is treated as missing, as in the original UCI file."""
    return pd.read_csv(path, na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data:
      - check all expected columns exist
      - drop duplicate rows
      - force all features to numeric (invalid values -> NaN)
      - convert 'num' (0-4) into a binary target (0 = no disease, 1 = disease)

    Missing values are NOT filled here on purpose: imputation happens inside
    the sklearn pipeline, so it is learned only from training data (no leakage).
    """
    missing_cols = set(FEATURES + [RAW_TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    df = df.copy().drop_duplicates()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    df[TARGET] = (df[RAW_TARGET] > 0).astype(int)
    return df[FEATURES + [TARGET]].reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    """Return X (features) and y (target)."""
    return df[FEATURES], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """
    Preprocessing pipeline:
      numeric  -> median impute + standard scaling
      nominal  -> mode impute + one-hot encoding (unknown categories ignored)
      binary   -> mode impute, kept as 0/1
      ordinal  -> mode impute, kept as 0-3 count
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    ordinal = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("nom", nominal, NOMINAL_FEATURES),
        ("bin", binary, BINARY_FEATURES),
        ("ord", ordinal, ORDINAL_FEATURES),
    ])


def main() -> None:
    df = clean_data(load_raw())
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)

    print(f"Cleaned data saved to: {CLEAN_FILE}")
    print(f"Shape: {df.shape}")
    print(f"Class counts:\n{df[TARGET].value_counts()}")
    print(f"Missing values left for the pipeline: {int(df.isna().sum().sum())}")

    # Inspection only: real training fits the preprocessor inside cross-validation
    X, _ = split_features_target(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    print(f"\nShape after preprocessing: {Xt.shape}")
    print("Output features:", list(pre.get_feature_names_out()))


if __name__ == "__main__":
    main()
