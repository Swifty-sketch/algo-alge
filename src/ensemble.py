"""
Ensemble model: XGBoost + Random Forest + (optional) LightGBM.
Trained on extended pro features. Predictions average all models;
agreement is the count of models above the buy threshold.
"""
import os
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.fetch_data import fetch
from src.features_pro import add_pro_features, make_pro_dataset, _PRO_DROP

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print('[ensemble] LightGBM not installed - using XGBoost + Random Forest only')


MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

XGB_PARAMS = {
    'n_estimators':     500,
    'max_depth':        5,
    'learning_rate':    0.04,
    'subsample':        0.85,
    'colsample_bytree': 0.85,
    'reg_alpha':        0.1,
    'reg_lambda':       0.5,
    'eval_metric':      'logloss',
    'random_state':     42,
}

LGB_PARAMS = {
    'n_estimators':     500,
    'max_depth':        5,
    'learning_rate':    0.04,
    'num_leaves':       31,
    'subsample':        0.85,
    'colsample_bytree': 0.85,
    'reg_alpha':        0.1,
    'reg_lambda':       0.5,
    'random_state':     42,
    'verbose':          -1,
}

RF_PARAMS = {
    'n_estimators':     300,
    'max_depth':        10,
    'min_samples_split': 50,
    'min_samples_leaf':  20,
    'random_state':      42,
    'n_jobs':            -1,
}


def _path(name):
    return os.path.join(MODELS_DIR, f'_pro_{name}.pkl')


def load_pro_models():
    models = {}
    for name in ('xgb', 'rf', 'lgb'):
        p = _path(name)
        if os.path.exists(p):
            try:
                models[name] = joblib.load(p)
            except Exception as e:
                print(f'[ensemble] failed to load {name}: {e}')
    return models


def load_feature_cols():
    p = _path('features')
    return joblib.load(p) if os.path.exists(p) else None


def train_pro(tickers, forward_days=3, threshold=0.015):
    total = len(tickers)
    print(f'\n=== TRAINING PRO ENSEMBLE ({total} stocks, {forward_days}-day forecast) ===')
    print(f'Step 1 of 3: Downloading data + extended feature engineering...\n')

    all_X, all_y = [], []
    for i, ticker in enumerate(tickers, 1):
        print(f'  [{i}/{total}] {ticker} - fetching...')
        df = fetch(ticker, period='5y', interval='1d', use_cache=True)
        if df is None or len(df) < 300:
            print(f'  [{i}/{total}] {ticker} - skipped (no data)')
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        try:
            X, y = make_pro_dataset(df, forward_days=forward_days, threshold=threshold)
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = y.loc[X.index]
            if len(X) >= 100:
                all_X.append(X)
                all_y.append(y)
                print(f'  [{i}/{total}] {ticker} - OK ({len(X)} rows)')
            else:
                print(f'  [{i}/{total}] {ticker} - skipped (only {len(X)} valid rows)')
        except Exception as e:
            print(f'  [{i}/{total}] {ticker} - error: {e}')

    if not all_X:
        print('[error] no data collected')
        return False

    X_all = pd.concat(all_X).reset_index(drop=True)
    y_all = pd.concat(all_y).reset_index(drop=True)
    print(f'\nStep 2 of 3: Training ensemble on {len(X_all):,} rows x {X_all.shape[1]} features')
    print(f'  Positive rate: {y_all.mean():.1%}')

    split = int(len(X_all) * 0.8)
    X_tr, X_te = X_all.iloc[:split], X_all.iloc[split:]
    y_tr, y_te = y_all.iloc[:split], y_all.iloc[split:]

    print('  >> Training XGBoost...')
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    auc_x = roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1])
    print(f'     XGBoost AUC: {auc_x:.3f}')

    print('  >> Training Random Forest...')
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    auc_r = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
    print(f'     Random Forest AUC: {auc_r:.3f}')

    lgb = None
    if HAS_LGB:
        print('  >> Training LightGBM...')
        lgb = LGBMClassifier(**LGB_PARAMS)
        lgb.fit(X_tr, y_tr)
        auc_l = roc_auc_score(y_te, lgb.predict_proba(X_te)[:, 1])
        print(f'     LightGBM AUC: {auc_l:.3f}')

    print(f'\nStep 3 of 3: Saving ensemble models...')
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(xgb, _path('xgb'))
    joblib.dump(rf,  _path('rf'))
    if lgb is not None:
        joblib.dump(lgb, _path('lgb'))
    joblib.dump(list(X_all.columns), _path('features'))

    print(f'=== PRO ENSEMBLE DONE ({2 + (1 if lgb else 0)} models saved) ===\n')
    return True


def predict_pro(X_row):
    """Returns dict with each model's prob, ensemble avg, and agreement label."""
    models = load_pro_models()
    if not models:
        return None

    probs = {}
    for name, m in models.items():
        try:
            probs[name] = float(m.predict_proba(X_row)[:, 1][0])
        except Exception as e:
            print(f'[predict_pro] {name} error: {e}')

    if not probs:
        return None

    ensemble = sum(probs.values()) / len(probs)
    buy_count = sum(1 for p in probs.values() if p >= 0.55)

    if buy_count == len(probs):
        agreement = 'STRONG'
    elif buy_count >= 2:
        agreement = 'MODERATE'
    elif buy_count == 1:
        agreement = 'WEAK'
    else:
        agreement = 'NONE'

    return {
        'ensemble':    round(ensemble, 3),
        'individual':  {k: round(v, 3) for k, v in probs.items()},
        'agreement':   agreement,
        'model_count': len(probs),
    }
