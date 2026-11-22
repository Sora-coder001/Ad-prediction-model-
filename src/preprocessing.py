import numpy as np
import pandas as pd

NEG1_COLS = ['previous_publisher_tenure_days', 'current_publisher_tenure_days',
             'cookie_age_days', 'session_elapsed_minutes', 'device_distinct_users_8w']
CAT_COLS = ['ad_format_code', 'audience_segment_code', 'ad_placement_code',
            'traffic_channel', 'device_platform']


def load_and_preprocess(train_path, test_path, sample=None):
    """Return X_p, T_p (NaNs kept), X_f, T_f (median-filled), y and test ids."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # drop rows with a missing target
    train = train.dropna(subset=['clicked']).reset_index(drop=True)
    if sample:
        train = train.sample(n=sample, random_state=42).reset_index(drop=True)

    y = train['clicked'].astype(int).values
    ids = test['impression_id'].values
    X = train.drop(columns=['clicked'])
    T = test.drop(columns=['impression_id'])
    if 'impression_id' in X.columns:
        X = X.drop(columns=['impression_id'])

    for df in (X, T):
        # flag each -1 "unavailable" value, then set it to NaN
        for c in NEG1_COLS:
            df[c + '_na'] = (df[c] == -1).astype(int)
            df[c] = df[c].replace(-1, np.nan)

        # engineered ratio and interaction features
        df['user_publisher_tenure_ratio'] = (
            df['current_publisher_tenure_days'].fillna(0) /
            (df['previous_publisher_tenure_days'].fillna(1) + 1)
        )
        df['session_intensity'] = (
            df['session_elapsed_minutes'] * df['device_distinct_users_8w']
        ).fillna(0)
        df['cookie_session_ratio'] = (
            df['cookie_age_days'] / (df['session_elapsed_minutes'].fillna(1) + 1)
        )

    # fill missing categoricals with the most frequent training value
    for c in CAT_COLS:
        mode = X[c].mode(dropna=True)[0]
        X[c] = X[c].fillna(mode)
        T[c] = T[c].fillna(mode)

    # one-hot encode categoricals on train and test together so columns match
    n = len(X)
    full = pd.get_dummies(pd.concat([X, T], axis=0).astype({c: str for c in CAT_COLS}),
                          columns=CAT_COLS, dtype=int)
    X_p, T_p = full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True)

    # median-filled copy for Random Forest and Extra Trees
    med = X_p.median()
    X_f, T_f = X_p.fillna(med), T_p.fillna(med)

    return X_p, T_p, X_f, T_f, y, ids
