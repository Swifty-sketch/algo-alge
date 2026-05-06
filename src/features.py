import pandas as pd
import numpy as np


def _atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close  = df['Close'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()
    open_  = df['Open'].squeeze()
    volume = df['Volume'].squeeze()

    # --- Trend / moving averages ---
    for n in [5, 10, 20, 50, 200]:
        sma = close.rolling(n).mean()
        df[f'sma_{n}']        = sma
        df[f'price_sma_{n}']  = close / sma - 1   # distance from SMA

    df['ema_12']      = close.ewm(span=12).mean()
    df['ema_26']      = close.ewm(span=26).mean()
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # --- Momentum ---
    for n in [1, 3, 5, 10, 20]:
        df[f'mom_{n}'] = close.pct_change(n)

    # --- RSI ---
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    rs    = (gain.ewm(alpha=1/14, min_periods=14).mean() /
             loss.ewm(alpha=1/14, min_periods=14).mean())
    df['rsi_14'] = 100 - 100 / (1 + rs)

    # --- Bollinger Bands ---
    sma20     = close.rolling(20).mean()
    std20     = close.rolling(20).std()
    bb_upper  = sma20 + 2 * std20
    bb_lower  = sma20 - 2 * std20
    df['bb_pct']   = (close - bb_lower) / (bb_upper - bb_lower)
    df['bb_width'] = (bb_upper - bb_lower) / sma20

    # --- Volatility ---
    df['atr_14']  = _atr(high, low, close, 14)
    df['atr_pct'] = df['atr_14'] / close
    df['vol_20d'] = close.pct_change().rolling(20).std()

    # --- Volume ---
    vol_sma20        = volume.rolling(20).mean()
    df['vol_ratio']  = volume / vol_sma20
    df['vol_change'] = volume.pct_change()

    # --- Candle shape ---
    df['body']        = (close - open_) / open_
    df['upper_wick']  = (high - close.clip(lower=open_)) / (close + 1e-9)
    df['lower_wick']  = (close.clip(upper=open_) - low)  / (close + 1e-9)

    # --- Calendar ---
    idx = pd.to_datetime(df.index)
    df['dow']   = idx.dayofweek
    df['month'] = idx.month

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def make_dataset(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02):
    df = add_features(df)

    future_ret  = df['Close'].squeeze().pct_change(forward_days).shift(-forward_days)
    df['target'] = (future_ret > threshold).astype(int)

    drop = {'Open', 'High', 'Low', 'Close', 'Volume',
            'Dividends', 'Stock Splits', 'target',
            'ema_12', 'ema_26',             # raw EMAs - keep derived only
            'sma_5','sma_10','sma_20','sma_50','sma_200'}
    feature_cols = [c for c in df.columns if c not in drop]

    df = df.dropna(subset=feature_cols + ['target'])
    return df[feature_cols], df['target']
