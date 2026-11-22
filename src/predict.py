import argparse
import json
import os

import numpy as np
import pandas as pd

from models import USES_FILLED, VARIANTS, make_model
from preprocessing import load_and_preprocess


def main():
    parser = argparse.ArgumentParser(description='Refit on all training data and write the submission file')
    parser.add_argument('--variant', choices=list(VARIANTS), default='ensemble_l1')
    parser.add_argument('--train', default='data/QM571_Train.csv')
    parser.add_argument('--test', default='data/QM571_Test_X.csv')
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--out', default='submissions')
    parser.add_argument('--sample', type=int, default=None, help='train on a row sample (quick check)')
    args = parser.parse_args()

    cfg = VARIANTS[args.variant]
    logreg_cfg = cfg['logreg']
    with open(os.path.join(args.artifacts, f'{args.variant}.json')) as f:
        saved = json.load(f)
    names = saved['names']
    weights = np.array([saved['weights'][k] for k in names])

    X_p, T_p, X_f, T_f, y, ids = load_and_preprocess(args.train, args.test, args.sample)

    test_preds = {}
    for name in names:
        Xa, Ta = (X_f, T_f) if name in USES_FILLED else (X_p, T_p)
        n_iter = int(saved['best_iters'][name] * 1.1) if name in saved['best_iters'] else None
        model = make_model(name, logreg_cfg, n_iter=n_iter)
        model.fit(Xa, y)
        test_preds[name] = model.predict_proba(Ta)[:, 1]
        print(f'{name} fitted on full train')

    # weighted average of predicted probabilities
    P = np.column_stack([test_preds[k] for k in names])
    final_pred = P @ weights

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'{args.variant}_submission.csv')
    submission = pd.DataFrame({'impression_id': ids, 'clicked': final_pred})
    submission.to_csv(path, index=False)
    print('Submission saved:', submission.shape, 'to', path)
    print('Prediction range:', final_pred.min().round(4), '-', final_pred.max().round(4))
    print(submission.head())


if __name__ == '__main__':
    main()
