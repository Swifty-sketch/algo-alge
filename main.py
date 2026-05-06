import sys
from config import (STOCKS, PERIOD, INTERVAL, FORWARD_DAYS, THRESHOLD,
                    TEST_SPLIT, XGB_PARAMS, INITIAL_CASH, COMMISSION,
                    ENTER_THRESH, EXIT_THRESH)
from src.fetch_data import fetch_all
from src.train import train, train_all
from src.backtest import run, run_all

USAGE = """
Usage:
  python main.py                         # fetch + train + backtest all stocks
  python main.py fetch                   # re-download all data (clears cache)
  python main.py train                   # train models for all stocks
  python main.py train  VOLV-B.ST        # train one stock
  python main.py backtest                # backtest all stocks
  python main.py backtest VOLV-B.ST      # backtest one stock
  python main.py all                     # full pipeline (uses cache)
"""

def main():
    args   = sys.argv[1:]
    cmd    = args[0].lower() if args else 'all'
    ticker = args[1].upper() if len(args) > 1 else None

    train_kwargs = dict(
        period=PERIOD, interval=INTERVAL,
        forward_days=FORWARD_DAYS, threshold=THRESHOLD,
        test_split=TEST_SPLIT, xgb_params=XGB_PARAMS,
    )
    bt_kwargs = dict(
        period=PERIOD, interval=INTERVAL,
        cash=INITIAL_CASH, commission=COMMISSION,
        enter_threshold=ENTER_THRESH, exit_threshold=EXIT_THRESH,
    )

    if cmd == 'fetch':
        fetch_all(STOCKS, period=PERIOD, interval=INTERVAL, use_cache=False)

    elif cmd == 'train':
        if ticker:
            train(ticker, use_cache=True, **train_kwargs)
        else:
            train_all(STOCKS, use_cache=True, **train_kwargs)

    elif cmd == 'backtest':
        if ticker:
            run(ticker, use_cache=True, **bt_kwargs)
        else:
            run_all(STOCKS, use_cache=True, **bt_kwargs)

    elif cmd == 'all':
        print('=== Step 1 / 3 - Fetch data ===')
        fetch_all(STOCKS, period=PERIOD, interval=INTERVAL, use_cache=True)
        print('\n=== Step 2 / 3 - Train models ===')
        train_all(STOCKS, use_cache=True, **train_kwargs)
        print('\n=== Step 3 / 3 - Backtest ===')
        run_all(STOCKS, use_cache=True, **bt_kwargs)

    else:
        print(USAGE)
        sys.exit(1)


if __name__ == '__main__':
    main()
