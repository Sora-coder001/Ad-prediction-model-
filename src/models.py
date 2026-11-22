import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

# each variant lists its models and the logistic regression penalty it uses
VARIANTS = {
    'ensemble_l1': {
        'description': 'RF, ET, LightGBM, XGBoost and L1 (lasso) logistic regression',
        'names': ['rf', 'et', 'lgb', 'xgb', 'logreg'],
        'logreg': {'penalty': 'l1', 'solver': 'saga'},
    },
    'ensemble_l2': {
        'description': 'RF, ET, LightGBM, XGBoost and L2 (ridge) logistic regression',
        'names': ['rf', 'et', 'lgb', 'xgb', 'logreg'],
        'logreg': {'penalty': 'l2', 'solver': 'lbfgs'},
    },
    'lean_lgb_logreg': {
        'description': 'LightGBM and L2 (ridge) logistic regression only',
        'names': ['lgb', 'logreg'],
        'logreg': {'penalty': 'l2', 'solver': 'lbfgs'},
    },
}

# models that train on the median-filled feature matrix
USES_FILLED = {'rf', 'et'}


def make_model(name, logreg_cfg, n_iter=None):
    """Build one model; n_iter=None uses early stopping, an int fixes the number of trees."""
    fixed = n_iter is not None
    if name == 'rf':
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features=0.4,
                                      n_jobs=-1, random_state=42)
    if name == 'et':
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=20, max_features=0.5,
                                    n_jobs=-1, random_state=42)
    if name == 'lgb':
        return lgb.LGBMClassifier(n_estimators=n_iter or 3000, learning_rate=0.02, num_leaves=8,
                                  min_child_samples=100, subsample=0.7, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=5, random_state=42, verbose=-1)
    if name == 'xgb':
        return xgb.XGBClassifier(n_estimators=n_iter or 3000, learning_rate=0.02, max_depth=3,
                                 min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                                 reg_lambda=5, eval_metric='auc',
                                 early_stopping_rounds=None if fixed else 100,
                                 random_state=42, n_jobs=-1)
    if name == 'logreg':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegressionCV(Cs=C_GRID, scoring='roc_auc', cv=3, max_iter=5000,
                                        n_jobs=-1, **logreg_cfg)),
        ])
    raise ValueError(f'unknown model: {name}')
