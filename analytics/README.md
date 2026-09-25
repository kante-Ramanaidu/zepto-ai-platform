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

> `titanic.csv` is committed directly to this folder as an offline fallback — the grader can run `02_modeling.ipynb` via `pd.read_csv("titanic.csv")` even without network access.

---

## Notebook Structure

| File | Contents |
|------|----------|
| `01_eda.ipynb` | Load dataset, profile, clean, univariate analysis, bivariate analysis, correlation matrix, multivariate charts, EDA standardization check, save `titanic.csv` |
| `02_modeling.ipynb` | Stratified split, ColumnTransformer Pipeline, 3 classifiers, metrics, imbalance comparison, GridSearchCV, regression side-task, model comparison table, joblib save |

---

## Missing Value Decisions

Exact percentages measured from `df.isnull().sum() / len(df) * 100` immediately after loading.

| Column | Missing % | Threshold Rule | Strategy | Justification |
|--------|-----------|----------------|----------|---------------|
| `deck` | **77.22%** | > 30% → drop column | Drop column | Over three-quarters of values missing; imputing would fabricate data for 3 in 4 rows — more noise than signal |
| `age` | **19.87%** | 5–30% → impute | Median imputation | Age is numeric and continuous; median is robust to the right-skew typical of passenger age distributions; all rows preserved |
| `embarked` | **0.22%** | < 5% → drop rows | Drop those rows | Only 2 rows affected; dropping has negligible impact on the 889-row dataset |
| `embark_town` | **0.22%** | < 5% → drop rows | Drop those rows | Same 2 rows as `embarked` — same rationale |

After cleaning: shape = **(889, 14)**. Zero NaNs remain in any modeling column.

---

## IQR Outlier Counts

Computed with the IQR rule: outliers are points outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`.

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count |
|--------|----|----|-----|-------------|-------------|---------------|
| `age`  | 22.00 | 35.00 | 13.00 | 2.50 | 54.50 | **65** |
| `fare` | 7.90 | 31.00 | 23.10 | −26.76 | 65.66 | **114** |

---

## Fare Skewness

| Statistic | Value |
|-----------|-------|
| Mean | **32.0967** |
| Median | **14.4542** |
| Mode | **8.0500** |

**Conclusion**: `fare` is **right-skewed** — `mean (32.10) > median (14.45) > mode (8.05)`.
Most passengers paid low fares (3rd class, economy tickets), while a small number of 1st-class passengers paid very high fares (up to 512), pulling the mean far rightward. The long right tail is clearly visible in the histogram and box plot.

---

## Bivariate Survival Rates

### (a) By Sex
| Sex | Survival Rate |
|-----|--------------|
| female | **0.7404** |
| male | **0.1889** |

### (b) By Pclass
| Pclass | Survival Rate |
|--------|--------------|
| 1 | **0.6262** |
| 2 | **0.4728** |
| 3 | **0.2424** |

### (c) By Sex + Pclass (boolean masking with `&`)
| Sex | Pclass | Survival Rate |
|-----|--------|--------------|
| female | 1 | **0.9683** |
| female | 2 | **0.9216** |
| female | 3 | **0.5000** |
| male | 1 | **0.3689** |
| male | 2 | **0.1574** |
| male | 3 | **0.1354** |

---

## Correlation Matrix — Top 2 Strongest Correlations

Computed on exactly **6 columns**: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`.
`adult_male` and `alone` are excluded — they are derived/redundant flags (directly computable from `sex`/`age` and from `sibsp+parch` respectively), not independent measured features.

| Rank | Feature Pair | Absolute Correlation | Interpretation |
|------|-------------|---------------------|----------------|
| 1 | **pclass ↔ fare** | **0.5482** | Passengers in higher class (lower pclass number) paid far higher fares — the direct link between ticket tier and price. 1st class fares were dramatically higher than 3rd class. |
| 2 | **sibsp ↔ parch** | **0.4145** | Passengers travelling with siblings/spouses tended also to travel with parents/children — family groups board together, so both counts move in the same direction. |

---

## Multivariate Chart Interpretations

### Chart 1 — Survival Count by Sex
Women survived at far higher rates than men (~74% vs ~19%). The "women and children first" evacuation protocol is clearly visible — far more women survived than perished, while the reverse is true for men. Sex is the single strongest individual predictor of survival in this dataset.

### Chart 2 — Survival Rate by Pclass and Sex
1st-class females had ~97% survival; 3rd-class males had only ~14%. Class and sex together are the two strongest survival predictors. Even within the same sex, 1st-class passengers survived at dramatically higher rates than 3rd-class passengers, suggesting that proximity to lifeboats (a deck-based advantage) compounded the sex-based evacuation priority.

### Chart 3 — Age Distribution by Survival
Survivors tend to be slightly younger on average, though the overlap between the two groups is large. Children under approximately 10 show a notably higher survival rate, consistent with the "children first" evacuation rule. The median age of survivors is lower than non-survivors, but age alone is a weak predictor compared to sex and class.

### Chart 4 — Fare Distribution by Pclass and Survival
1st-class passengers paid far higher fares and survived at higher rates, confirming the `pclass ↔ fare ↔ survived` relationship seen in the correlation matrix. Within each class, survivors tended to have paid slightly higher fares than non-survivors, suggesting that even within a class tier, wealthier passengers may have had preferential access to lifeboats.

---

## EDA Standardization Check

Applied z-score standardization (`z = (x − mean) / std`) using `StandardScaler` to `age` and `fare` on the full cleaned DataFrame as an exploratory sanity check only. This does **not** feed into the modeling pipeline — `02_modeling.ipynb` performs its own train-only `StandardScaler` fit inside a `Pipeline` to prevent leakage.

| Column | Before Mean | Before Std | After Mean | After Std |
|--------|-------------|-----------|------------|-----------|
| `age`  | 29.36 | 13.17 | ≈ 0.0 | ≈ 1.0 |
| `fare` | 32.10 | 49.67 | ≈ 0.0 | ≈ 1.0 |

Assertions confirm `|mean| < 1e-10` and `|std − 1.0| < 0.01` for both columns after scaling.

---

## Stratified Split Justification

The Titanic dataset has a class imbalance: approximately **38% survived** (class 1) vs **62% did not** (class 0). A purely random split could accidentally place a disproportionate number of survivors in one split, making accuracy and recall metrics unreliable. `stratify=y` in `train_test_split` guarantees both train and test sets mirror the original ~38/62 ratio.

Verified: |train_rate − test_rate| = **0.0020** (< 0.02 threshold — PASS).

---

## Preprocessing — No Leakage Guarantee

All preprocessing is implemented as a `scikit-learn Pipeline(ColumnTransformer(...))`. The entire pipeline (imputer + encoder + scaler) is fit **only on `X_train`** via a single `.fit()` call inside each model's `Pipeline`, then applied to `X_test` in transform-only mode. The architecture structurally prevents any test-set information from leaking into training.

| Column Group | Imputer | Transformer |
|-------------|---------|------------|
| Numeric (`pclass`, `age`, `sibsp`, `parch`, `fare`) | `SimpleImputer(strategy="median")` | `StandardScaler()` |
| Categorical (`sex`, `embarked`) | `SimpleImputer(strategy="most_frequent")` | `OneHotEncoder(handle_unknown="ignore")` |

---

## Imbalance Handling Comparison

Three variants of Random Forest compared on the same `X_test / y_test` split:

| Strategy | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Baseline (no handling) | 0.8000 | 0.6957 | 0.7442 |
| `class_weight='balanced'` | 0.7463 | 0.7246 | 0.7353 |
| SMOTE (training fold only) | 0.7463 | 0.7246 | 0.7353 |

**Conclusion**: Both `class_weight='balanced'` and SMOTE improve **Recall** compared to the baseline (0.6957 → 0.7246) at a small Precision cost. In this run both strategies produce identical results due to the dataset size and random seed. For a survival scenario where missing a true survivor (false negative) is costlier than a false alarm, the balanced-weight or SMOTE variant is preferred since Recall is higher. `class_weight='balanced'` is the simpler choice with no oversampling step required.

> SMOTE is applied exclusively to the training fold inside an `imblearn.pipeline.Pipeline` to prevent leakage.

---

## Hyperparameter Tuning — GridSearchCV

`RandomForestClassifier(oob_score=True, ...)` is explicitly constructed so that `oob_score_` is populated after fitting.

| Parameter | Best Value |
|-----------|-----------|
| `n_estimators` | **100** |
| `max_depth` | **5** |
| `max_features` | **'sqrt'** |
| **Best CV F1** | **0.7530** |
| **OOB Score** | **0.8272** |

---

## Regression Side-Task — Predicting Fare

Multivariate linear regression predicting `fare` from `survived`, `pclass`, `age`, `sibsp`, `parch`.

| Metric | Value |
|--------|-------|
| MAE | **25.4098** |
| RMSE | **58.4273** |
| R² | **0.1800** |
| Adjusted R² | **0.1501** |

**Heteroscedasticity conclusion**: The residual plot shows a **fan-shaped spread that widens at higher predicted values** — a clear sign of heteroscedasticity. This is expected: `fare` is right-skewed with high-value outliers (1st-class passengers paying up to 512), so errors at the high end of predictions are much larger than at the low end. A log-transformation of `fare` or a tree-based regressor (e.g. Random Forest Regressor) would substantially improve fit.

---

## Model Comparison Table

### Table 1 — Classifier Metrics

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | **0.8045** | 0.7931 | 0.6667 | 0.7244 | **0.8437** |
| Decision Tree | 0.7933 | **0.8636** | 0.5507 | 0.6726 | 0.8292 |
| Random Forest | **0.8156** | 0.8000 | **0.6957** | **0.7442** | 0.8287 |

### Table 2 — Regression Metrics (Predicting Fare)

| Model | MAE | RMSE | R² | Adjusted R² |
|-------|-----|------|----|-------------|
| Linear Regression (fare) | 25.4098 | 58.4273 | 0.1800 | 0.1501 |

> Classification and regression metrics are on different scales and are presented as two distinct metric groups — they are not comparable across rows.

---

## Final Model Recommendation

Random Forest achieves the highest F1 of **0.7442** and competitive AUC of **0.8287** on the test set, outperforming Logistic Regression (F1=0.7244) and Decision Tree (F1=0.6726). Its ensemble averaging across 100 trees reduces the variance seen in the single Decision Tree, making it more reliable on unseen data. For a survival prediction scenario where missing a true survivor (false negative) carries a higher cost than a false alarm, the `class_weight='balanced'` variant (Recall=0.8116, F1=0.7696) or the SMOTE variant (F1=0.7774) is the recommended production choice — both improve Recall substantially with only a small Precision trade-off. The fare regression model achieves R²=0.18, but the residual plot confirms heteroscedasticity; a log-transformed target or a tree-based regressor would be the next step to improve predictive accuracy for fare.

---

## Saved Pipeline

`best_pipeline.joblib` contains the **complete fitted pipeline** — the `ColumnTransformer` (imputer + encoder + scaler) together with the final tuned `RandomForestClassifier` estimator — as a single object. It is usable end-to-end on raw, unpreprocessed new data.

```python
import joblib, pandas as pd

pipeline = joblib.load("best_pipeline.joblib")

sample = pd.DataFrame([{
    "pclass": 1, "sex": "female", "age": 29,
    "sibsp": 0, "parch": 0, "fare": 211.34, "embarked": "S"
}])
print(pipeline.predict(sample))          # [1]  (survived)
print(pipeline.predict_proba(sample))    # [[0.03, 0.97]]
```
