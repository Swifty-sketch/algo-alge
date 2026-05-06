import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from src.fetch_data import fetch
from src.features import add_features, make_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_FEAT_DROP = {
    'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
    'ema_12', 'ema_26', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
}

SWING_CFG    = {'forward_days': 3,   'threshold': 0.015}
LONGTERM_CFG = {'forward_days': 252, 'threshold': 0.20}

XGB_PARAMS = {
    'n_estimators':     400,
    'max_depth':        5,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'eval_metric':      'logloss',
    'random_state':     42,
}


def _model_path(model_type):
    return os.path.join(MODELS_DIR, f'_universal_{model_type}.pkl')


def load_universal(model_type):
    p = _model_path(model_type)
    return joblib.load(p) if os.path.exists(p) else None


def train_universal(tickers, model_type='swing', forward_days=3, threshold=0.015):
    total = len(tickers)
    print(f'\n=== TRAINING {model_type.upper()} MODEL ({total} stocks, {forward_days}-day forecast) ===')
    print(f'Step 1 of 2: Downloading & processing stock data...\n')

    all_X, all_y = [], []
    for i, ticker in enumerate(tickers, 1):
        print(f'  [{i}/{total}] {ticker} — fetching data...')
        df = fetch(ticker, period='5y', interval='1d', use_cache=True)
        if df is None or len(df) < 300:
            print(f'  [{i}/{total}] {ticker} — skipped (no data)')
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        try:
            X, y = make_dataset(df, forward_days=forward_days, threshold=threshold)
            X    = X.replace([np.inf, -np.inf], np.nan).dropna()
            y    = y.loc[X.index]
            if len(X) >= 100:
                all_X.append(X)
                all_y.append(y)
                print(f'  [{i}/{total}] {ticker} — OK ({len(X)} rows)')
            else:
                print(f'  [{i}/{total}] {ticker} — skipped (only {len(X)} valid rows)')
        except Exception as e:
            print(f'  [{i}/{total}] {ticker} — error: {e}')

    if not all_X:
        print('[error] no data collected — aborting')
        return None

    X_all = pd.concat(all_X).reset_index(drop=True)
    y_all = pd.concat(all_y).reset_index(drop=True)
    good  = len(all_X)
    print(f'\nStep 2 of 2: Training XGBoost on {good} stocks ({len(X_all):,} rows)...')
    print(f'  Positive rate: {y_all.mean():.1%}  |  Please wait...')

    split       = int(len(X_all) * 0.8)
    X_tr, X_te  = X_all.iloc[:split], X_all.iloc[split:]
    y_tr, y_te  = y_all.iloc[:split], y_all.iloc[split:]

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    proba = model.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)
    print(f'  AUC: {auc:.3f}')

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, _model_path(model_type))
    print(f'[saved] {model_type} model saved successfully')
    print(f'=== {model_type.upper()} MODEL DONE ===\n')
    return model


def _get_signal(ticker, model):
    df = fetch(ticker, period='5y', interval='1d', use_cache=True)
    if df is None or len(df) < 250:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    feat_df   = add_features(df)
    feat_cols = [c for c in feat_df.columns if c not in _FEAT_DROP]
    X = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        return None

    prob      = float(model.predict_proba(X.iloc[[-1]])[:, 1][0])
    close     = df['Close'].squeeze()
    price     = round(float(close.iloc[-1]), 2)
    change_1d = round(float(close.pct_change().iloc[-1]) * 100, 2)
    return {'ticker': ticker, 'signal': round(prob, 3),
            'price': price, 'change_1d': change_1d}


def scan(tickers, model_type='swing'):
    model = load_universal(model_type)
    if model is None:
        print(f'[scan] no {model_type} model — train first')
        return []

    results = []
    for ticker in tickers:
        try:
            sig = _get_signal(ticker, model)
            if sig:
                results.append(sig)
        except Exception as e:
            print(f'[scan] {ticker}: {e}')

    results.sort(key=lambda x: -x['signal'])
    return results
