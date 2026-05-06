import os
import pandas as pd
from backtesting import Backtest

from src.fetch_data import fetch
from src.strategy import attach_signals, MLStrategy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def run(ticker, period='5y', interval='1d',
        cash=100_000, commission=0.002,
        enter_threshold=0.60, exit_threshold=0.40,
        use_cache=True, plot=True):

    print(f'\n[backtest] {ticker}')
    df = fetch(ticker, period=period, interval=interval, use_cache=use_cache)
    if df is None:
        return None

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    try:
        df = attach_signals(df, ticker)
    except FileNotFoundError as e:
        print(f'[skip] {e}')
        return None

    class Strategy(MLStrategy):
        pass
    Strategy.enter_threshold = enter_threshold
    Strategy.exit_threshold  = exit_threshold

    bt    = Backtest(df, Strategy, cash=cash, commission=commission,
                     exclusive_orders=True)
    stats = bt.run()

    keys = ['Return [%]', 'Buy & Hold Return [%]', 'Sharpe Ratio',
            'Max. Drawdown [%]', 'Win Rate [%]', '# Trades']
    print(stats[keys].to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if plot:
        safe     = ticker.replace('/', '_').replace(':', '_')
        out_path = os.path.join(RESULTS_DIR, f'{safe}.html')
        bt.plot(filename=out_path, open_browser=False)
        print(f'  [plot] {out_path}')

    return stats


def run_all(stocks, **kwargs):
    summary = {}
    for ticker in stocks:
        stats = run(ticker, **kwargs)
        if stats is not None:
            summary[ticker] = {
                'return_%':   round(stats['Return [%]'], 2),
                'bnh_%':      round(stats['Buy & Hold Return [%]'], 2),
                'sharpe':     round(stats['Sharpe Ratio'], 3),
                'max_dd_%':   round(stats['Max. Drawdown [%]'], 2),
                'win_rate_%': round(stats['Win Rate [%]'], 2),
                'trades':     stats['# Trades'],
            }

    if not summary:
        print('[warn] no results to summarize')
        return None

    df_sum = pd.DataFrame(summary).T.sort_values('sharpe', ascending=False)
    print('\n=== SUMMARY ===')
    print(df_sum.to_string())

    out = os.path.join(RESULTS_DIR, 'summary.csv')
    df_sum.to_csv(out)
    print(f'\n[saved] {out}')
    return df_sum
