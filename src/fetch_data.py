import os
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def fetch(ticker, period='5y', interval='1d', use_cache=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    safe = ticker.replace('/', '_').replace(':', '_')
    cache_path = os.path.join(DATA_DIR, f'{safe}_{period}_{interval}.csv')

    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f'[cache] {ticker}: {len(df)} rows')
        return df

    print(f'[fetch] {ticker} ...')
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)

    if df is None or df.empty:
        print(f'[warn]  {ticker}: no data returned')
        return None

    # flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    print(f'[fetch] {ticker}: {len(df)} rows saved')
    return df


def fetch_all(stocks, period='5y', interval='1d', use_cache=True):
    results = {}
    for ticker in stocks:
        df = fetch(ticker, period=period, interval=interval, use_cache=use_cache)
        if df is not None:
            results[ticker] = df
    return results
