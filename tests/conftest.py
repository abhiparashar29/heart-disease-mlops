"""Shared pytest fixtures."""
import joblib
import pytest

from src.preprocessing import RAW_FILE, ROOT, clean_data, load_raw

MODEL_PATH = ROOT / "models" / "model.joblib"


@pytest.fixture(scope="session")
def raw_df():
    return load_raw(RAW_FILE)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture
def sample_patient():
    """First patient of the dataset, as the API will receive it."""
    return {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
    }


@pytest.fixture(scope="session")
def trained_model():
    if not MODEL_PATH.exists():
        pytest.skip("models/model.joblib not found - run `python -m src.train` first")
    return joblib.load(MODEL_PATH)
