# Report Notes – Heart Disease MLOps

## 1. Data & EDA
- See notebooks/01_eda.ipynb (observations section) and reports/figures/.
- Missing values: ca (4), thal (2) → imputed inside the pipeline (mode), not dropped.

## 2. Feature Engineering
- 13 raw features → 22 model features after preprocessing.
- Numeric (age, trestbps, chol, thalach, oldpeak): median impute + StandardScaler.
- Nominal (cp, restecg, slope, thal): mode impute + One-Hot Encoding (handle_unknown="ignore").
- Binary (sex, fbs, exang) and ordinal (ca): mode impute, kept as-is.
- All steps in one sklearn ColumnTransformer inside the model Pipeline → identical
  transforms at training, testing and serving time; no data leakage.

## 3. Model Development & Tuning
- Split: 80% train (242) / 20% test (61), stratified, random_state = 42.
- Tuning: GridSearchCV, 5-fold stratified CV, selection metric = ROC-AUC.
- Candidates: Logistic Regression (16 combos), Random Forest (36), Gradient Boosting (8).

| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Recall |
|---|---|---|---|---|
| Logistic Regression | 0.905 ± 0.019 | 0.960 | 0.869 | 0.929 |
| Random Forest | 0.901 ± 0.024 | 0.958 | 0.869 | 0.929 |
| Gradient Boosting | 0.874 ± 0.028 | 0.946 | 0.885 | 0.929 |

**Selected model:** Logistic Regression — C = 1.0, L2 penalty, no class weighting.

**Why:**
- Highest CV ROC-AUC and lowest variance across folds (most stable).
- Statistically tied with Random Forest → chose the simpler, interpretable model
  (coefficients explain risk factors — valuable in a medical setting), smaller
  and faster to serve.
- Selected on CV score, not test score, to keep the test set an unbiased check.
- Balanced class weights did not improve results — consistent with the mild
  54/46 class balance seen in EDA.
- Test recall 0.929: 26 of 28 diseased patients correctly identified.
- Test scores > CV scores because the test set is small (61 rows) → higher variance;
  CV is the more reliable estimate.

## 4. Experiment Tracking (MLflow)
- Experiment: heart-disease-classification
- 3 parent runs (one per model) + 60 nested child runs (one per hyperparameter combination).
- Logged: parameters, CV metrics (mean ± std), test metrics, confusion matrix,
  ROC curve, feature importance, classification report, grid results CSV,
  model with signature + input example.
- Best run tagged selected_as_best = true.
- Screenshots: screenshots/mlflow_*.png

## 5. Model Packaging & Reproducibility
- models/model.joblib — full pipeline (preprocessing + model).
- models/metadata.json — model name, MLflow run ID, params, metrics, feature list,
  training timestamp, sklearn & Python versions.
- requirements.txt with pinned versions; fixed random_state everywhere.

## 6. CI/CD  (to do)
## 7. Docker  (to do)
## 8. Kubernetes deployment  (to do)
## 9. Monitoring  (to do)
## 10. Architecture diagram  (to do)
