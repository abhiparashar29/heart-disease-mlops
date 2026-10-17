"""
Download the UCI Heart Disease (Cleveland) dataset and save it as a raw CSV.

Usage:
    python src/download_data.py

Output:
    data/raw/heart_raw.csv
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_FILE = RAW_DIR / "heart_raw.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

DIRECT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def download_via_ucimlrepo() -> pd.DataFrame:
    """Primary method: official UCI Python package (dataset id 45)."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=45)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return df[COLUMNS]


def download_direct() -> pd.DataFrame:
    """Fallback: read the raw file directly. Missing values are '?'."""
    return pd.read_csv(DIRECT_URL, header=None, names=COLUMNS, na_values="?")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = download_via_ucimlrepo()
        source = "ucimlrepo"
    except Exception as exc:
        print(f"ucimlrepo failed ({exc}); falling back to direct URL")
        df = download_direct()
        source = "direct URL"

    df.to_csv(RAW_FILE, index=False)

    print(f"Downloaded via {source}")
    print(f"Saved to: {RAW_FILE}")
    print(f"Shape: {df.shape}")
    print("Missing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()
