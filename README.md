# Ad Click Prediction

Predicts whether an ad impression results in a click. The competition metric is ROC-AUC and the submission contains predicted probabilities.

The training data has 60,000 rows and 35 columns. The click rate is 1.35%, so the target is heavily imbalanced.

## Project layout

```
data/        QM571_Train.csv, QM571_Test_X.csv
src/         preprocessing.py, models.py, train.py, predict.py
notebooks/   ad_click_prediction.ipynb (same pipeline in separate cells)
```

## Preprocessing

- Rows with a missing target are dropped.
- In the five columns where -1 means "unavailable", each -1 gets an `_na` flag and is then set to NaN.
- Three engineered features: `user_publisher_tenure_ratio`, `session_intensity`, `cookie_session_ratio`.
- Missing categorical values are filled with the most frequent value.
- Categoricals are one-hot encoded on train and test together so the columns match.
- Random Forest and Extra Trees use a median-filled copy; LightGBM and XGBoost keep NaN; logistic regression uses median imputation and standard scaling.

## Model versions

| Variant | Models | Logistic regression | Weighted OOF AUC |
|---|---|---|---|
| `ensemble_l1` | RF, ET, LightGBM, XGBoost, LogReg | L1 (lasso, saga) | 0.79803 |
| `ensemble_l2` | RF, ET, LightGBM, XGBoost, LogReg | L2 (ridge, lbfgs) | 0.79876 |
| `lean_lgb_logreg` | LightGBM, LogReg | L2 (ridge, lbfgs) | run `train.py` to reproduce |

Each model is scored with stratified 5-fold cross-validation. Blend weights are found with Nelder-Mead on the out-of-fold AUC, and the final prediction is the weighted average of raw probabilities. The final models are refit on all training data, with LightGBM and XGBoost using the mean best iteration across folds x 1.1.

## Usage

```bash
pip install -r requirements.txt

# 1. cross-validate, print per-model AUC and weights, save artifacts/<variant>.json
python src/train.py --variant ensemble_l2

# 2. refit on all data and write submissions/<variant>_submission.csv
python src/predict.py --variant ensemble_l2
```

Variants: `ensemble_l1`, `ensemble_l2`, `lean_lgb_logreg`. Add `--sample 6000` for a quick check on a subset.

The notebook runs the same steps in separate cells; set `VARIANT` in the model fitting cell.
