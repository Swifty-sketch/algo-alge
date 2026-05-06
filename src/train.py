import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.fetch_data import fetch
from src.features import make_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def train(ticker, period='5y', interval='1d',
          forward_days=5, threshold=0.02, test_split=0.2,
          xgb_params=None, use_cache=True, save=True):

    print(f'\n[train] {ticker}')
    df = fetch(ticker, period=period, interval=interval, use_cache=use_cache)
    if df is None or len(df) < 300:
        print(f'[skip]  {ticker}: not enough data ({len(df) if df is not None else 0} rows)')
        return None

    X, y = make_dataset(df, forward_days=forward_days, threshold=threshold)
    if len(X) < 200:
        print(f'[skip]  {ticker}: not enough feature rows after dropna')
        return None

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    if len(X) < 200:
        print(f'[skip]  {ticker}: not enough rows after inf removal')
        return None

    split      = int(len(X) * (1 - test_split))
    X_train    = X.iloc[:split]
    X_test     = X.iloc[split:]
    y_train    = y.iloc[:split]
    y_test     = y.iloc[split:]

    params = xgb_params or {
        'n_estimators':     300,
        'max_depth':        4,
        'learning_rate':    0.05,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'eval_metric':      'logloss',
        'random_state':     42,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)

    print(classification_report(y_test, preds,
                                 target_names=['Down/Flat', 'Up 2%+'],
                                 digits=3))
    print(f'  AUC: {auc:.3f}  |  test rows: {len(y_test)}  |  '
          f'base rate: {y_test.mean():.1%}')

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        safe = ticker.replace('/', '_').replace(':', '_')
        path = os.path.join(MODELS_DIR, f'{safe}.pkl')
        joblib.dump(model, path)
        print(f'  [saved] {path}')

    return model, auc


def train_all(stocks, **kwargs):
    results = {}
    for ticker in stocks:
        out = train(ticker, **kwargs)
        if out:
            model, auc = out
            results[ticker] = {'model': model, 'auc': auc}
    return results


def load_model(ticker):
    safe = ticker.replace('/', '_').replace(':', '_')
    path = os.path.join(MODELS_DIR, f'{safe}.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)
