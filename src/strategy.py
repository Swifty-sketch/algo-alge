import os
import numpy as np
import pandas as pd
import joblib
from backtesting import Strategy

from src.features import add_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def attach_signals(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    safe = ticker.replace('/', '_').replace(':', '_')
    model_path = os.path.join(MODELS_DIR, f'{safe}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No trained model for {ticker}. Run: python main.py train {ticker}')

    model = joblib.load(model_path)

    feat_df = add_features(df)
    drop    = {'Open', 'High', 'Low', 'Close', 'Volume',
               'Dividends', 'Stock Splits',
               'ema_12', 'ema_26',
               'sma_5','sma_10','sma_20','sma_50','sma_200'}
    feat_cols = [c for c in feat_df.columns if c not in drop]

    X      = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
    proba  = model.predict_proba(X)[:, 1]

    signal = pd.Series(0.0, index=df.index)
    signal.loc[X.index] = proba

    df = df.copy()
    df['Signal'] = signal
    return df


class MLStrategy(Strategy):
    enter_threshold = 0.60
    exit_threshold  = 0.40

    def init(self):
        self.signal = self.I(lambda: self.data.Signal, name='ML prob')

    def next(self):
        sig = self.signal[-1]
        if not self.position:
            if sig >= self.enter_threshold:
                self.buy()
        else:
            if sig < self.exit_threshold:
                self.position.close()
