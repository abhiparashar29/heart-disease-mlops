"""
Train, tune and compare classifiers for heart disease prediction.
Every run is tracked in MLflow; the best pipeline is saved for serving.

Usage (from repo root):
    python -m src.train
    mlflow ui --backend-store-uri mlruns --port 5000
"""
import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # no screen needed (works in CI / Docker)
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.preprocessing import (  # noqa: E402
    FEATURES,
    ROOT,
    build_preprocessor,
    clean_data,
    load_raw,
    split_features_target,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
EXPERIMENT_NAME = "heart-disease-classification"

MODELS_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "reports" / "training"

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
REFIT_METRIC = "roc_auc"

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 3, 5],
            "model__max_features": ["sqrt", 0.5],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    ),
}


def make_pipeline(model) -> Pipeline:
    """Preprocessing + model in one object, so serving uses identical transforms."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def evaluate(pipeline, X, y):
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    metrics = {
        "test_accuracy": accuracy_score(y, y_pred),
        "test_precision": precision_score(y, y_pred),
        "test_recall": recall_score(y, y_pred),
        "test_f1": f1_score(y, y_pred),
        "test_roc_auc": roc_auc_score(y, y_proba),
    }
    return metrics, y_pred, y_proba


def save_eval_plots(name, y_true, y_pred, y_proba):
    out = ARTIFACT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax
    )
    ax.set_title(f"{name}: confusion matrix (test set)")
    cm_path = out / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title(f"{name}: ROC curve (test set)")
    roc_path = out / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    return [cm_path, roc_path]


def save_feature_importance(pipeline, name):
    """Tree models: feature_importances_. Logistic Regression: coefficients."""
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Importance"
    else:
        values, label = model.coef_[0], "Coefficient"

    series = pd.Series(values, index=[n.split("__", 1)[1] for n in feature_names])
    series = series.reindex(series.abs().sort_values().index).tail(15)

    fig, ax = plt.subplots(figsize=(7, 6))
    series.plot(kind="barh", ax=ax)
    ax.set_title(f"{name}: top features")
    ax.set_xlabel(label)
    path = ARTIFACT_DIR / name / "feature_importance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def log_grid_candidates(search, name):
    """One nested MLflow run per hyperparameter combination tried."""
    results = pd.DataFrame(search.cv_results_)
    for i, row in results.iterrows():
        with mlflow.start_run(run_name=f"{name}_candidate_{i:02d}", nested=True):
            params = {
                col.replace("param_model__", ""): row[col]
                for col in results.columns
                if col.startswith("param_")
            }
            mlflow.log_params(params)
            mlflow.log_metrics({f"cv_{m}": row[f"mean_test_{m}"] for m in SCORING})
            mlflow.set_tag("cv_rank", int(row[f"rank_test_{REFIT_METRIC}"]))
    return results


def main() -> None:
    df = clean_data(load_raw())
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    summary, best = [], None

    for name, (model, grid) in CANDIDATES.items():
        print(f"\n=== Tuning {name} ===")
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(
                make_pipeline(model), grid, scoring=SCORING,
                refit=REFIT_METRIC, cv=cv, n_jobs=-1,
            )
            search.fit(X_train, y_train)
            results = log_grid_candidates(search, name)

            idx = search.best_index_
            cv_metrics = {}
            for m in SCORING:
                cv_metrics[f"cv_{m}_mean"] = float(search.cv_results_[f"mean_test_{m}"][idx])
                cv_metrics[f"cv_{m}_std"] = float(search.cv_results_[f"std_test_{m}"][idx])

            best_pipeline = search.best_estimator_
            test_metrics, y_pred, y_proba = evaluate(best_pipeline, X_test, y_test)

            mlflow.set_tags({
                "model_type": name,
                "dataset": "UCI Heart Disease (Cleveland)",
                "selection_metric": f"cv_{REFIT_METRIC}",
            })
            mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_params({
                "cv_folds": CV_FOLDS,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_candidates": len(results),
                "n_train": len(X_train),
                "n_test": len(X_test),
            })
            mlflow.log_metrics({**cv_metrics, **test_metrics})

            plots = save_eval_plots(name, y_test, y_pred, y_proba)
            plots.append(save_feature_importance(best_pipeline, name))
            for p in plots:
                mlflow.log_artifact(str(p), artifact_path="plots")

            report_path = ARTIFACT_DIR / name / "classification_report.txt"
            report_path.write_text(
                classification_report(y_test, y_pred, target_names=["No disease", "Disease"])
            )
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            grid_path = ARTIFACT_DIR / name / "grid_search_results.csv"
            results.to_csv(grid_path, index=False)
            mlflow.log_artifact(str(grid_path), artifact_path="reports")

            signature = infer_signature(X_train, best_pipeline.predict_proba(X_train))
            mlflow.sklearn.log_model(
                best_pipeline, artifact_path="model",
                signature=signature, input_example=X_train.head(3),
            )

            row = {"model": name, "run_id": run.info.run_id, **cv_metrics, **test_metrics}
            summary.append(row)
            if best is None or cv_metrics["cv_roc_auc_mean"] > best["cv_roc_auc_mean"]:
                best = {**row, "pipeline": best_pipeline, "params": search.best_params_}

            print(f"best params: {search.best_params_}")
            print(f"CV ROC-AUC: {cv_metrics['cv_roc_auc_mean']:.3f} "
                  f"± {cv_metrics['cv_roc_auc_std']:.3f} | "
                  f"Test ROC-AUC: {test_metrics['test_roc_auc']:.3f} | "
                  f"Test recall: {test_metrics['test_recall']:.3f}")

    comparison = pd.DataFrame(summary).drop(columns="run_id")
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    print("\n=== Model comparison ===")
    print(comparison.round(3).to_string(index=False))

    # Save the winning pipeline (preprocessing + model) for the API
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODELS_DIR / "model.joblib")
    metadata = {
        "model_name": best["model"],
        "mlflow_run_id": best["run_id"],
        "best_params": {k.replace("model__", ""): str(v) for k, v in best["params"].items()},
        "cv_roc_auc_mean": best["cv_roc_auc_mean"],
        "test_metrics": {k: float(v) for k, v in best.items() if k.startswith("test_")},
        "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected_as_best", "true")

    print(f"\nBest model: {best['model']} (CV ROC-AUC {best['cv_roc_auc_mean']:.3f})")
    print(f"Saved: {MODELS_DIR / 'model.joblib'} and metadata.json")


if __name__ == "__main__":
    main()