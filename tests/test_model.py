"""Unit tests for model training and the saved model artifact."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.preprocessing import FEATURES, ROOT, split_features_target
from src.train import make_pipeline


def to_frame(patient: dict) -> pd.DataFrame:
    return pd.DataFrame([patient])[FEATURES].astype(float)


def test_pipeline_trains_and_beats_baseline(clean_df):
    """A freshly trained pipeline must clearly beat random guessing (AUC 0.5)."""
    X, y = split_features_target(clean_df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = make_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    assert auc > 0.80


def test_saved_model_predicts_valid_output(trained_model, sample_patient):
    X = to_frame(sample_patient)
    pred = trained_model.predict(X)
    proba = trained_model.predict_proba(X)
    assert pred[0] in (0, 1)
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_saved_model_handles_missing_values(trained_model, sample_patient):
    patient = {**sample_patient, "ca": np.nan, "thal": np.nan}
    proba = trained_model.predict_proba(to_frame(patient))
    assert not np.isnan(proba).any()


def test_high_risk_patient_scores_higher_than_low_risk(trained_model):
    """Sanity check based on EDA findings."""
    high_risk = {
        "age": 65, "sex": 1, "cp": 4, "trestbps": 160, "chol": 300, "fbs": 1,
        "restecg": 2, "thalach": 110, "exang": 1, "oldpeak": 3.0,
        "slope": 2, "ca": 3, "thal": 7,
    }
    low_risk = {
        "age": 40, "sex": 0, "cp": 3, "trestbps": 115, "chol": 200, "fbs": 0,
        "restecg": 0, "thalach": 180, "exang": 0, "oldpeak": 0.0,
        "slope": 1, "ca": 0, "thal": 3,
    }
    p_high = trained_model.predict_proba(to_frame(high_risk))[0, 1]
    p_low = trained_model.predict_proba(to_frame(low_risk))[0, 1]
    assert p_high > p_low


def test_model_is_deterministic(trained_model, clean_df):
    X, _ = split_features_target(clean_df)
    first = trained_model.predict_proba(X.head(20))
    second = trained_model.predict_proba(X.head(20))
    np.testing.assert_array_equal(first, second)


def test_metadata_is_consistent():
    meta_path = ROOT / "models" / "metadata.json"
    if not meta_path.exists():
        pytest.skip("metadata.json not found")
    meta = json.loads(meta_path.read_text())
    assert meta["features"] == FEATURES
    assert meta["test_metrics"]["test_roc_auc"] > 0.80
