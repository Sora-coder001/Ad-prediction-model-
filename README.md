# Ad Click Prediction

Predicting whether an online ad impression will be clicked, built for a class competition scored on **ROC-AUC**. The submission is a predicted click probability for each impression in the test set.

## Problem

Every row is one ad impression: who saw it (cookie, device and publisher history), where it ran (placement, format, traffic channel) and the session it happened in. The target `clicked` is 1 if the user clicked and 0 otherwise.

The main challenge is imbalance. Only **1.35%** of training impressions were clicked, so a model that always predicts "no click" scores 98.65% accuracy and is useless. That is why the competition uses ROC-AUC, which measures how well the model ranks clicked impressions above non-clicked ones, and why the output is probabilities rather than 0/1 labels.

## Data

| File | Rows | Notes |
|---|---|---|
| `data/QM571_Train.csv` | 60,000 | 35 columns including the `clicked` target |
| `data/QM571_Test_X.csv` | 20,000 | Same features plus `impression_id`, no target |

There are two kinds of missing values:
- Five numeric columns use **-1** to mean "unavailable": `previous_publisher_tenure_days`, `current_publisher_tenure_days`, `cookie_age_days`, `session_elapsed_minutes` and `device_distinct_users_8w`.
- Five categorical columns have blanks: `ad_format_code`, `audience_segment_code`, `ad_placement_code`, `traffic_channel` and `device_platform`.

## Approach

### 1. Preprocessing (`src/preprocessing.py`)

- **Drop rows with a missing target.**
- **Handle -1 sentinels.** Each -1 gets its own `_na` flag column and is then replaced with NaN. Because "this value is unavailable" can itself predict clicks, the flag keeps that signal instead of treating -1 as a real small number.
- **Engineer three features:**
  - `user_publisher_tenure_ratio`: current publisher tenure relative to previous tenure
  - `session_intensity`: session length × number of distinct users on the device
  - `cookie_session_ratio`: cookie age relative to session length
- **Impute categoricals** with the most frequent training value.
- **One-hot encode** the categoricals on train and test together, so both have identical columns.
- **Make model-specific copies:**
  - LightGBM and XGBoost receive the data with NaN kept, since they handle missing values natively.
  - Random Forest and Extra Trees receive a median-filled copy.
  - Logistic regression applies median imputation and standard scaling inside its own pipeline.

### 2. Models (`src/models.py`)

| Model | Family | Why it is included |
|---|---|---|
| Random Forest | Bagged trees | Stable baseline with low variance |
| Extra Trees | Bagged trees | More randomised splits than RF, so it adds diversity |
| LightGBM | Gradient boosting | Strongest single model on this data |
| XGBoost | Gradient boosting | A second boosting implementation |
| Logistic Regression | Linear | Makes different errors from the tree models, which helps the blend |

LightGBM and XGBoost are kept deliberately shallow and regularised (learning rate 0.02, small trees, row/column subsampling, L2 penalty), with early stopping on validation AUC. Logistic regression tunes its regularisation strength `C` via `LogisticRegressionCV` over a grid from 0.001 to 10, using either an L1 (lasso) or L2 (ridge) penalty.

### 3. Validation and blending (`src/train.py`)

- **Stratified 5-fold cross-validation** keeps the 1.35% click rate the same in every fold.
- Every model produces **out-of-fold (OOF) predictions**: each training row is scored by a model that never saw it, so all models are compared fairly on the same rows.
- **Blend weights** come from Nelder-Mead optimisation, which searches for the weights that maximise OOF ROC-AUC. A model that adds nothing new can end up with a weight close to zero.
- The final prediction is a **weighted average of raw probabilities**, as the submission requires.

### 4. Final fit and submission (`src/predict.py`)

Each model is refit on all 60,000 training rows. LightGBM and XGBoost use the average best iteration from cross-validation × 1.1, which accounts for having more data. The weighted blend of test probabilities is written to `submissions/<variant>_submission.csv` with columns `impression_id, clicked`.

## Results

Individual model OOF ROC-AUC (5-fold CV):

| Model | OOF AUC |
|---|---|
| Random Forest | 0.752 |
| Extra Trees | 0.765 |
| Logistic Regression (ridge) | 0.783 |
| XGBoost | 0.795 |
| LightGBM | 0.797 |

Weighted ensembles:

| Variant | Models | Logistic regression | Weighted OOF AUC |
|---|---|---|---|
| `ensemble_l2` | RF, ET, LightGBM, XGBoost, LogReg | L2 (ridge) | **0.79876** |
| `ensemble_l1` | RF, ET, LightGBM, XGBoost, LogReg | L1 (lasso) | 0.79803 |
| `lean_lgb_logreg` | LightGBM, LogReg | L2 (ridge) | run `train.py` to reproduce |

## Key takeaways

- **Gradient boosting was the strongest single model** on this imbalanced, mixed-type tabular data.
- **Ensembling helps only when the models differ.** LightGBM and XGBoost are similar enough that the optimiser often gives one of them little weight. Logistic regression is weaker alone but earns real weight because its errors differ from the trees'.
- **Missing values can carry signal.** Flagging -1 sentinels before imputing preserved information that plain imputation would lose.
- **ROC-AUC fits imbalanced problems.** It judges how well clicks are ranked, not raw accuracy, which a 1.35% positive rate makes misleading.

## Project structure

```
Ad-prediction-model-/
├── data/
│   ├── QM571_Train.csv
│   └── QM571_Test_X.csv
├── notebooks/
│   └── ad_click_prediction.ipynb   # full pipeline in separate cells (Colab-friendly)
├── src/
│   ├── preprocessing.py            # cleaning, feature engineering, encoding
│   ├── models.py                   # model definitions and the three variants
│   ├── train.py                    # 5-fold CV, per-model AUC, blend weights
│   └── predict.py                  # refit on full data, write submission
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# 1. Cross-validate: prints per-model AUC, weights and ensemble AUC,
#    and saves them to artifacts/<variant>.json
python src/train.py --variant ensemble_l2

# 2. Refit on all training data and write submissions/<variant>_submission.csv
python src/predict.py --variant ensemble_l2
```

- Variants: `ensemble_l1`, `ensemble_l2`, `lean_lgb_logreg`
- Add `--sample 6000` to either command for a quick test run on a subset.
- **Notebook:** open `notebooks/ad_click_prediction.ipynb` and run the cells in order: imports → preprocessing → model fitting → submission. Set `VARIANT` in the model fitting cell. In Colab, upload the two CSVs and set `DATA_DIR = '.'`.

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · SciPy
