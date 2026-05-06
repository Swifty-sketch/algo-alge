"""
Extended feature set for the pro model.
Adds 20+ indicators on top of basic features:
trend strength, advanced momentum, volume flow, volatility, multi-timeframe.
"""
import numpy as np
import pandas as pd

from src.features import add_features


def add_pro_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)

    close  = df['Close'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()
    open_  = df['Open'].squeeze()
    volume = df['Volume'].squeeze()

    # ── Stochastic Oscillator (14, 3) ──
    n = 14
    low_n  = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rng    = (high_n - low_n).replace(0, np.nan)
    df['stoch_k'] = 100 * (close - low_n) / rng
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ── Williams %R ──
    df['williams_r'] = -100 * (high_n - close) / rng

    # ── CCI (20) ──
    tp = (high + low + close) / 3
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['cci'] = (tp - ma) / (0.015 * md.replace(0, np.nan))

    # ── ADX (14) ──
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr       = pd.concat([high - low,
                          (high - close.shift()).abs(),
                          (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr      = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr.replace(0, np.nan)
    di_sum   = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / di_sum
    df['adx']      = dx.ewm(alpha=1/14, min_periods=14).mean()
    df['plus_di']  = plus_di
    df['minus_di'] = minus_di

    # ── On-Balance Volume + slope ──
    obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    df['obv_slope_10'] = obv.diff(10) / obv.rolling(10).mean().abs().replace(0, np.nan)

    # ── Money Flow Index (14) ──
    raw_money = tp * volume
    pos_flow  = raw_money.where(tp > tp.shift(), 0).rolling(14).sum()
    neg_flow  = raw_money.where(tp < tp.shift(), 0).rolling(14).sum()
    mfi_ratio = pos_flow / neg_flow.replace(0, np.nan)
    df['mfi'] = 100 - 100 / (1 + mfi_ratio)

    # ── Rate of change at multiple horizons ──
    for horizon in (5, 10, 20, 50, 100):
        df[f'roc_{horizon}'] = close.pct_change(horizon) * 100

    # ── Volume features ──
    df['vol_roc_5']    = volume.pct_change(5)
    df['vol_roc_20']   = volume.pct_change(20)
    df['vol_ratio_50'] = volume / volume.rolling(50).mean().replace(0, np.nan)

    # ── Volatility ratios ──
    df['atr_ratio']    = atr / atr.rolling(60).mean().replace(0, np.nan)
    df['return_std_5'] = close.pct_change().rolling(5).std()

    # ── 52-week distance (252 trading days) ──
    high_52w = close.rolling(252).max()
    low_52w  = close.rolling(252).min()
    df['dist_from_52w_high'] = (close - high_52w) / high_52w * 100
    df['dist_from_52w_low']  = (close - low_52w)  / low_52w.replace(0, np.nan) * 100

    # ── Moving-average crosses ──
    if {'sma_5','sma_20'}.issubset(df.columns):
        df['cross_5_20']   = (df['sma_5']  > df['sma_20']).astype(int)
    if {'sma_20','sma_50'}.issubset(df.columns):
        df['cross_20_50']  = (df['sma_20'] > df['sma_50']).astype(int)
    if {'sma_50','sma_200'}.issubset(df.columns):
        df['cross_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)

    # ── Candlestick patterns ──
    body  = (close - open_).abs()
    rng2  = (high - low).replace(0, np.nan)
    df['body_to_range'] = body / rng2
    df['upper_shadow']  = (high - df[['Close','Open']].max(axis=1)) / rng2
    df['lower_shadow']  = (df[['Close','Open']].min(axis=1) - low)  / rng2
    df['is_doji']       = (body / rng2 < 0.1).astype(int)

    # ── Gap from previous close ──
    df['gap_pct'] = (open_ - close.shift()) / close.shift() * 100

    # ── Statistical (return distribution) ──
    rets = close.pct_change()
    df['skew_20'] = rets.rolling(20).skew()
    df['kurt_20'] = rets.rolling(20).kurt()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


_PRO_DROP = {
    'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
    'ema_12', 'ema_26', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
}


def make_pro_dataset(df: pd.DataFrame, forward_days: int = 3, threshold: float = 0.015):
    df = add_pro_features(df)

    fwd = df['Close'].squeeze().pct_change(forward_days).shift(-forward_days)
    y   = (fwd > threshold).astype(int)

    feat_cols = [c for c in df.columns if c not in _PRO_DROP]
    X = df[feat_cols].dropna()
    y = y.loc[X.index]
    return X, y
