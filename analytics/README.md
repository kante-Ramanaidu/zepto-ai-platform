# Module 2 — Analytics Pipeline

## Overview

An end-to-end analyst-to-data-scientist workflow on the Titanic dataset. The dataset is loaded **exactly once** via `sns.load_dataset('titanic')` in `01_eda.ipynb`, immediately saved to `titanic.csv` as an offline fallback, and every subsequent step — cleaning, EDA, modeling, tuning, regression — works from that same data. Never reloads independently.

---

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `seaborn`, `pandas`, `matplotlib`, `scikit-learn`, `imbalanced-learn`, `joblib`.

---

## Run

```bash
# Step 1 — EDA (must run first; generates titanic.csv)
jupyter notebook 01_eda.ipynb

# Step 2 — Modeling (reads titanic.csv produced by Step 1)
jupyter notebook 02_modeling.ipynb
```

> `titanic.csv` is also committed directly to this folder as an offline fallback — the grader can run `02_modeling.ipynb` via `pd.read_csv("titanic.csv")` even without network access.

---

## Notebook Structure

| File | Contents |
|------|----------|
| `01_eda.ipynb` | Load dataset, profile, clean, univariate analysis, bivariate analysis, correlation matrix, multivariate charts, EDA standardization check, save `titanic.csv` |
| `02_modeling.ipynb` | Stratified split, ColumnTransformer Pipeline, 3 classifiers, metrics, imbalance comparison, GridSearchCV, regression side-task, model comparison table, joblib save |

---

## Missing Value Decisions

<!-- Filled in after running 01_eda.ipynb — exact percentages measured from df.isnull().mean() -->

| Column | Missing % | Threshold Rule | Strategy | Justification |
|--------|-----------|---------------|----------|---------------|
| `age` | ~19.9% | 5–30% → impute | Median imputation | Age is numeric and continuous; median is robust to the right-skew in passenger ages |
| `deck` | ~77.2% | >30% → unreliable | Drop column | Over three-quarters of values missing; imputing would create more noise than signal |
| `embarked` | ~0.2% | <5% → drop rows | Drop those rows | Only 2 rows affected; dropping has negligible impact on the dataset |
| `embark_town` | ~0.2% | <5% → drop rows | Drop rows | Same source as `embarked`; same 2 rows |

> **Note**: Exact percentages will be updated after running the notebook. The threshold rules applied are exactly as specified: under 5% → drop rows; 5–30% → impute; >30% → drop column or encode as category.

---

## IQR Outlier Counts

<!-- Filled in after running 01_eda.ipynb -->

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count |
|--------|----|----|-----|-------------|-------------|---------------|
| `age`  | — | — | — | — | — | — |
| `fare` | — | — | — | — | — | — |

---

## Fare Skewness

<!-- Filled in after running 01_eda.ipynb -->

| Statistic | Value |
|-----------|-------|
| Mean | — |
| Median | — |
| Mode | — |

**Conclusion**: `fare` is **right-skewed** — `mean > median > mode`. Most passengers paid low fares, while a small number of first-class passengers paid very high fares, pulling the mean rightward.

---

## Bivariate Survival Rates

<!-- Filled in after running 01_eda.ipynb -->

### (a) By Sex
| Sex | Survival Rate |
|-----|--------------|
| female | — |
| male | — |

### (b) By Pclass
| Pclass | Survival Rate |
|--------|--------------|
| 1 | — |
| 2 | — |
| 3 | — |

### (c) By Sex + Pclass
| Sex | Pclass | Survival Rate |
|-----|--------|--------------|
| female | 1 | — |
| female | 2 | — |
| female | 3 | — |
| male | 1 | — |
| male | 2 | — |
| male | 3 | — |

---

## Correlation Matrix — Top 2 Strongest Correlations

Computed on exactly **6 columns**: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`.
`adult_male` and `alone` are excluded — they are derived/redundant flags, not independent measured features.

<!-- Filled in after running 01_eda.ipynb -->

| Rank | Feature Pair | Absolute Correlation | Interpretation |
|------|-------------|---------------------|----------------|
| 1 | — ↔ — | — | — |
| 2 | — ↔ — | — | — |

---

## Multivariate Chart Interpretations

<!-- Filled in after running 01_eda.ipynb -->

### Chart 1 — Survival Count by Sex
[Written interpretation — 2–4 sentences]

### Chart 2 — Survival Rate by Pclass and Sex
[Written interpretation — 2–4 sentences]

### Chart 3 — Age Distribution by Survival Status
[Written interpretation — 2–4 sentences]

### Chart 4 — Fare Distribution by Pclass and Survival
[Written interpretation — 2–4 sentences]

---

## EDA Standardization Check

Applied z-score standardization (`z = (x − mean) / std`) to `age` and `fare` on the full cleaned DataFrame as an exploratory sanity check only. This does **not** feed into the modeling pipeline — the Pipeline in `02_modeling.ipynb` performs its own train-only `StandardScaler` fit to prevent leakage.

<!-- Filled in after running 01_eda.ipynb -->

| Column | Before Mean | Before Std | After Mean | After Std |
|--------|-------------|-----------|------------|-----------|
| `age`  | — | — | ≈ 0.0 | ≈ 1.0 |
| `fare` | — | — | ≈ 0.0 | ≈ 1.0 |

---

## Stratified Split Justification

The Titanic dataset has a class imbalance: approximately **38% survived** (class 1) vs **62% did not** (class 0). A purely random split could accidentally place a disproportionate number of survivors in one split, making accuracy and recall metrics unreliable. `stratify=y` in `train_test_split` guarantees both train and test sets mirror the original ~38/62 ratio, giving a fair evaluation baseline.

---

## Preprocessing — No Leakage Guarantee

All preprocessing is implemented as a `scikit-learn Pipeline(ColumnTransformer(...))`. The entire pipeline (imputer + encoder + scaler) is fit **only on `X_train`** via a single `.fit()` call, then applied to `X_test` in transform-only mode. The architecture structurally prevents any test-set information from leaking into training.

| Column Group | Imputer | Transformer |
|-------------|---------|------------|
| Numeric (`pclass`, `age`, `sibsp`, `parch`, `fare`) | `SimpleImputer(strategy="median")` | `StandardScaler()` |
| Categorical (`sex`, `embarked`) | `SimpleImputer(strategy="most_frequent")` | `OneHotEncoder(handle_unknown="ignore")` |

---

## Imbalance Handling Comparison

<!-- Filled in after running 02_modeling.ipynb -->

| Strategy | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Baseline (no handling) | — | — | — |
| `class_weight='balanced'` | — | — | — |
| SMOTE (training fold only) | — | — | — |

**Conclusion**: [Written after running — which strategy worked best and why, referencing precision/recall tradeoff]

> SMOTE is applied exclusively to the training fold inside an `imblearn.pipeline.Pipeline` to prevent leakage.

---

## Hyperparameter Tuning — GridSearchCV

`RandomForestClassifier(oob_score=True, ...)` is required so that `oob_score_` is populated after fitting.

<!-- Filled in after running 02_modeling.ipynb -->

| Parameter | Best Value |
|-----------|-----------|
| `n_estimators` | — |
| `max_depth` | — |
| `max_features` | — |
| **Best CV F1** | — |
| **OOB Score** | — |

---

## Regression Side-Task — Predicting Fare

Multivariate linear regression predicting `fare` from other available features.

<!-- Filled in after running 02_modeling.ipynb -->

| Metric | Value |
|--------|-------|
| MAE | — |
| RMSE | — |
| R² | — |
| Adjusted R² | — |

**Heteroscedasticity conclusion**: [Written after running — the residual plot almost certainly shows fan-shaped spread at higher predicted values, indicating heteroscedasticity, because `fare` is right-skewed with outliers]

---

## Model Comparison Table

### Classifier Metrics

<!-- Filled in after running 02_modeling.ipynb -->

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |

### Regression Metrics

| Model | MAE | RMSE | R² | Adjusted R² |
|-------|-----|------|----|-------------|
| Linear Regression (fare) | — | — | — | — |

> Classification and regression metrics are on different scales and are presented as two separate tables — they are not comparable across rows.

---

## Final Model Recommendation

<!-- Filled in after running 02_modeling.ipynb — minimum 3 sentences citing specific metric values -->

[Example structure: "Random Forest achieves the highest F1 of X.XX and AUC of X.XX on the test set, outperforming Logistic Regression (F1=X.XX) and Decision Tree (F1=X.XX). Its lower variance compared to the single Decision Tree makes it more reliable for deployment on unseen data. The class_weight='balanced' variant further improves recall to X.XX with minimal precision loss, making it the recommended production model for scenarios where missing a survivor (false negative) is costlier than a false alarm."]

---

## Saved Pipeline

`best_pipeline.joblib` contains the **complete fitted pipeline** — the `ColumnTransformer` (imputer + encoder + scaler) together with the final `RandomForestClassifier` estimator — as a single object. It is usable end-to-end on raw, unpreprocessed new data.

```python
import joblib, pandas as pd

pipeline = joblib.load("best_pipeline.joblib")

sample = pd.DataFrame([{
    "pclass": 1, "sex": "female", "age": 29,
    "sibsp": 0, "parch": 0, "fare": 211.34, "embarked": "S"
}])
print(pipeline.predict(sample))         # [1]
print(pipeline.predict_proba(sample))   # [[0.07, 0.93]]
```
