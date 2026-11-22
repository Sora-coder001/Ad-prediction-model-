import argparse
import json
import os

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from models import USES_FILLED, VARIANTS, make_model
from preprocessing import load_and_preprocess


def main():
    parser = argparse.ArgumentParser(description='5-fold CV, AUC per model and blend weights')
    parser.add_argument('--variant', choices=list(VARIANTS), default='ensemble_l1')
    parser.add_argument('--train', default='data/QM571_Train.csv')
    parser.add_argument('--test', default='data/QM571_Test_X.csv')
    parser.add_argument('--out', default='artifacts')
    parser.add_argument('--sample', type=int, default=None, help='train on a row sample (quick check)')
    args = parser.parse_args()

    cfg = VARIANTS[args.variant]
    names, logreg_cfg = cfg['names'], cfg['logreg']
    X_p, T_p, X_f, T_f, y, ids = load_and_preprocess(args.train, args.test, args.sample)
    n = len(X_p)
    print('Train:', X_p.shape, '| Test:', T_p.shape, '| Click rate:', round(y.mean(), 4))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = {k: np.zeros(n) for k in names}
    best_iters = {k: [] for k in ['lgb', 'xgb'] if k in names}

    for fold, (tr, va) in enumerate(skf.split(X_p, y)):
        for name in names:
            Xa = X_f if name in USES_FILLED else X_p
            Xtr, Xva, ytr, yva = Xa.iloc[tr], Xa.iloc[va], y[tr], y[va]
            model = make_model(name, logreg_cfg)
            if name == 'lgb':
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                best_iters[name].append(model.best_iteration_)
            elif name == 'xgb':
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                best_iters[name].append(model.best_iteration + 1)
            else:
                model.fit(Xtr, ytr)
            oof[name][va] = model.predict_proba(Xva)[:, 1]
        print(f'Fold {fold + 1} done')

    # blend weights that maximize out-of-fold AUC on raw probabilities
    O = np.column_stack([oof[k] for k in names])

    def neg_auc(w):
        w = np.abs(w) / np.abs(w).sum()
        return -roc_auc_score(y, O @ w)

    res = minimize(neg_auc, np.ones(len(names)) / len(names), method='Nelder-Mead',
                   options={'maxiter': 2000})
    weights = np.abs(res.x) / np.abs(res.x).sum()

    print('\nPer-model results:')
    for idx, k in enumerate(names):
        print(f'{k}: OOF AUC = {roc_auc_score(y, oof[k]):.5f}, Weight = {weights[idx]:.3f}')
    print(f'\nEqual-weight ensemble OOF AUC: {roc_auc_score(y, O.mean(axis=1)):.5f}')
    print(f'Weighted ensemble OOF AUC: {-res.fun:.5f}')
    print('Avg best iterations:', {k: int(np.mean(v)) for k, v in best_iters.items()})

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'{args.variant}.json')
    with open(path, 'w') as f:
        json.dump({'variant': args.variant,
                   'names': names,
                   'weights': dict(zip(names, weights.tolist())),
                   'best_iters': {k: int(np.mean(v)) for k, v in best_iters.items()},
                   'oof_auc': {k: float(roc_auc_score(y, oof[k])) for k in names},
                   'ensemble_oof_auc': float(-res.fun)}, f, indent=2)
    print('Saved weights to', path)


if __name__ == '__main__':
    main()
