import os, sys, threading, secrets
from functools import wraps
import numpy as np
from datetime import datetime

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import (STOCKS, PERIOD, INTERVAL, FORWARD_DAYS, THRESHOLD,
                    TEST_SPLIT, XGB_PARAMS, INITIAL_CASH, COMMISSION,
                    ENTER_THRESH, EXIT_THRESH)
from src.fetch_data import fetch
from src.features import add_features
from src.features_pro import add_pro_features
from src.train import train_all, load_model
from src.backtest import run_all
from src.scanner import train_universal, scan, load_universal, SWING_CFG, LONGTERM_CFG
from src.ensemble import train_pro, predict_pro, load_pro_models, load_feature_cols
from src.universe import get_sp100, get_sp500, get_all_listed, get_reddit_tickers

app = Flask(__name__)
CORS(app)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), 'models')

_task = {'running': False, 'message': 'Idle', 'updated': None}
_log  = []

_FEAT_DROP = {
    'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
    'ema_12', 'ema_26', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
}

_NEWS_POS = {
    'surge', 'soar', 'beat', 'record', 'upgrade', 'rally', 'gain', 'rise', 'jump',
    'strong', 'outperform', 'bullish', 'profit', 'raises', 'raised', 'upside',
    'breakthrough', 'approved', 'approval', 'wins', 'expands', 'deal', 'buyback',
    'dividend', 'growth', 'acquire', 'acquisition', 'partnership', 'top', 'best',
}
_NEWS_NEG = {
    'fall', 'drop', 'miss', 'cut', 'downgrade', 'decline', 'loss', 'weak',
    'underperform', 'bearish', 'warning', 'lawsuit', 'investigation', 'crash',
    'plunge', 'slump', 'concern', 'delay', 'recall', 'fraud', 'fine', 'layoffs',
}


class _LogCapture:
    def write(self, text):
        try:
            sys.__stdout__.write(text)
        except UnicodeEncodeError:
            safe = text.encode(sys.__stdout__.encoding or 'utf-8', errors='replace') \
                       .decode(sys.__stdout__.encoding or 'utf-8', errors='replace')
            sys.__stdout__.write(safe)
        line = text.rstrip()
        if line:
            _log.append(line)
            if len(_log) > 1000:
                del _log[0]
    def flush(self):
        sys.__stdout__.flush()


def _run_task(fn):
    global _task, _log
    _log  = []
    _task = {'running': True, 'message': 'Running...', 'updated': datetime.now().isoformat()}
    old_stdout = sys.stdout
    sys.stdout = _LogCapture()
    try:
        fn()
        _task = {'running': False, 'message': 'Done', 'updated': datetime.now().isoformat()}
    except Exception as e:
        _task = {'running': False, 'message': f'Error: {e}', 'updated': datetime.now().isoformat()}
        print(f'ERROR: {e}')
    finally:
        sys.stdout = old_stdout


def _get_proba(model, df):
    feat_df   = add_features(df)
    feat_cols = [c for c in feat_df.columns if c not in _FEAT_DROP]
    X = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        return None, None
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index), feat_cols


# ── Swedish stocks (my stocks) ─────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file('dashboard.html')


@app.route('/api/signals')
def get_signals():
    results = []
    for ticker in STOCKS:
        model = load_model(ticker)
        entry = {'ticker': ticker, 'trained': model is not None,
                 'signal': None, 'price': None, 'change_1d': None, 'error': None}
        if model:
            try:
                df = fetch(ticker, period='5y', interval='1d', use_cache=True)
                if df is not None and len(df) > 60:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    proba_s, _ = _get_proba(model, df)
                    if proba_s is not None and len(proba_s):
                        close = df['Close'].squeeze()
                        entry['signal']    = round(float(proba_s.iloc[-1]), 3)
                        entry['price']     = round(float(close.iloc[-1]), 2)
                        entry['change_1d'] = round(float(close.pct_change().iloc[-1]) * 100, 2)
                    else:
                        entry['error'] = 'No valid feature rows'
            except Exception as e:
                entry['error'] = str(e)
        results.append(entry)
    return jsonify(results)


@app.route('/api/summary')
def get_summary():
    path = os.path.join(RESULTS_DIR, 'summary.csv')
    if not os.path.exists(path):
        return jsonify([])
    df = pd.read_csv(path, index_col=0)
    return jsonify(df.reset_index().rename(columns={'index': 'ticker'}).to_dict(orient='records'))


@app.route('/api/chart/<path:ticker>')
def get_chart(ticker):
    ticker = ticker.upper()
    df = fetch(ticker, period='2y', interval='1d', use_cache=True)
    if df is None:
        return jsonify({'error': 'No data'}), 404
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].copy()
    df.index = pd.to_datetime(df.index)

    probs = []
    # try per-stock model first, then universal swing
    model = load_model(ticker) or load_universal('swing')
    if model:
        full = fetch(ticker, period='2y', interval='1d', use_cache=True)
        if isinstance(full.columns, pd.MultiIndex):
            full.columns = full.columns.get_level_values(0)
        proba_s, _ = _get_proba(model, full)
        if proba_s is not None:
            aligned = proba_s.reindex(df.index).ffill().fillna(0)
            probs   = [round(float(x), 3) for x in aligned]

    dates  = df.index.strftime('%Y-%m-%d').tolist()
    prices = [round(float(x), 2) for x in df['Close']]
    return jsonify({'dates': dates, 'prices': prices, 'probabilities': probs})


@app.route('/api/task-status')
def task_status():
    return jsonify(_task)


@app.route('/api/log')
def get_log():
    since = int(request.args.get('since', 0))
    return jsonify({'lines': _log[since:], 'total': len(_log)})


@app.route('/api/train', methods=['POST'])
def trigger_train():
    if _task['running']:
        return jsonify({'error': 'Already running'}), 409
    kw = dict(period=PERIOD, interval=INTERVAL, forward_days=FORWARD_DAYS,
              threshold=THRESHOLD, test_split=TEST_SPLIT, xgb_params=XGB_PARAMS,
              use_cache=True, save=True)
    threading.Thread(target=_run_task,
                     args=(lambda: train_all(STOCKS, **kw),),
                     daemon=True).start()
    return jsonify({'ok': True})


@app.route('/api/backtest', methods=['POST'])
def trigger_backtest():
    if _task['running']:
        return jsonify({'error': 'Already running'}), 409
    kw = dict(period=PERIOD, interval=INTERVAL, cash=INITIAL_CASH,
              commission=COMMISSION, enter_threshold=ENTER_THRESH,
              exit_threshold=EXIT_THRESH, use_cache=True, plot=True)
    threading.Thread(target=_run_task,
                     args=(lambda: run_all(STOCKS, **kw),),
                     daemon=True).start()
    return jsonify({'ok': True})


# ── US market scanner ──────────────────────────────────────────────────────────

@app.route('/api/train-universal', methods=['POST'])
def trigger_train_universal():
    if _task['running']:
        return jsonify({'error': 'Already running'}), 409
    tickers = get_sp500()

    def _train_both():
        train_universal(tickers, 'swing',    **SWING_CFG)
        train_universal(tickers, 'longterm', **LONGTERM_CFG)

    threading.Thread(target=_run_task, args=(_train_both,), daemon=True).start()
    return jsonify({'ok': True})


@app.route('/api/scan/<model_type>')
def api_scan(model_type):
    if model_type not in ('swing', 'longterm'):
        return jsonify({'error': 'Unknown model type'}), 400

    sp500 = get_sp500()

    # pull reddit mentions — 15 subs, 100 posts each + comments
    reddit_raw = []
    try:
        reddit_raw = get_reddit_tickers(limit=100)
    except Exception as e:
        print(f'[reddit] fetch error: {e}')

    reddit_tickers = [t for t, _ in reddit_raw]
    sp500_set      = set(sp500)
    reddit_set     = set(reddit_tickers)

    # union: S&P 500 base + everything Reddit is talking about
    all_tickers = list(dict.fromkeys(sp500 + reddit_tickers))
    print(f'[scan] scanning {len(all_tickers)} stocks ({len(sp500)} S&P500 + {len(reddit_tickers)} Reddit)')

    results = scan(all_tickers, model_type)

    for r in results:
        sources = []
        if r['ticker'] in sp500_set:
            sources.append('S&P 500')
        if r['ticker'] in reddit_set:
            sources.append('Reddit')
        r['source'] = ', '.join(sources) if sources else 'Other'

    return jsonify(results)


@app.route('/api/reddit')
def api_reddit():
    try:
        tickers = get_reddit_tickers(limit=100)
        return jsonify([{'ticker': t, 'mentions': c} for t, c in tickers])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/universal-status')
def universal_status():
    return jsonify({
        'swing':    os.path.exists(os.path.join(MODELS_DIR, '_universal_swing.pkl')),
        'longterm': os.path.exists(os.path.join(MODELS_DIR, '_universal_longterm.pkl')),
    })


# ── Penny / hype stock finder ──────────────────────────────────────────────────

_PENNY_SUBS = ['pennystocks', 'smallstreetbets', 'RobinHoodPennyStocks',
               'wallstreetbets', 'stocks']
_PENNY_HEADERS = {'User-Agent': 'trading-algo/1.0'}


def _scrape_penny_reddit(limit=30):
    import re, requests as req
    ticker_re  = re.compile(r'(?<!\w)\$?([A-Z]{2,5})(?!\w)')
    blacklist  = {'A','I','OR','FOR','ARE','THE','AN','AT','IN','ON','UP','GO',
                  'IT','BE','BY','TO','DO','OF','IF','US','ALL','NEW','NOW','HOW',
                  'BIG','LOW','HIGH','BUY','SELL','HOLD','DD','TA','IMO','YOLO',
                  'WSB','ETF','IPO','CEO','USD','SPY','QQQ','IWM','VIX','EV','AI'}
    mentions   = {}
    posts_by_t = {}

    for sub in _PENNY_SUBS:
        try:
            url  = f'https://www.reddit.com/r/{sub}/hot.json?limit={limit}'
            resp = req.get(url, headers=_PENNY_HEADERS, timeout=8)
            if resp.status_code != 200:
                continue
            for post in resp.json()['data']['children']:
                d     = post['data']
                title = d.get('title', '')
                body  = d.get('selftext', '')[:500]
                text  = title + ' ' + body
                found = set()
                for m in ticker_re.finditer(text):
                    t = m.group(1)
                    if t not in blacklist and 2 <= len(t) <= 5:
                        found.add(t)
                for t in found:
                    mentions[t]   = mentions.get(t, 0) + 1
                    if t not in posts_by_t:
                        posts_by_t[t] = []
                    if len(posts_by_t[t]) < 4:
                        posts_by_t[t].append(title)
        except Exception as e:
            print(f'[penny reddit] {sub}: {e}')

    return mentions, posts_by_t


def _make_recommendation(signal, vol_ratio, mom_5d, market_cap):
    score    = 0
    reasons  = []
    warnings = []

    if signal is not None:
        if signal >= 0.70:
            score += 3
            reasons.append(f'ML model is {round(signal*100)}% confident in near-term upside')
        elif signal >= 0.55:
            score += 1
            reasons.append(f'ML model shows mild bullish pattern ({round(signal*100)}%)')
        else:
            score -= 2
            warnings.append(f'ML model is bearish ({round(signal*100)}% — below 55%)')

    if vol_ratio is not None:
        if vol_ratio >= 4:
            score += 3
            reasons.append(f'Volume is {vol_ratio:.1f}x above average — heavy unusual buying')
        elif vol_ratio >= 2:
            score += 2
            reasons.append(f'Volume is {vol_ratio:.1f}x above average — elevated interest')
        elif vol_ratio >= 1.3:
            score += 1
            reasons.append(f'Volume slightly above average ({vol_ratio:.1f}x)')
        elif vol_ratio < 0.5:
            score -= 1
            warnings.append('Volume is very low — low liquidity risk')

    if mom_5d is not None:
        if mom_5d >= 0.15:
            score += 2
            reasons.append(f'Strong 5-day momentum (+{mom_5d:.0%})')
        elif mom_5d >= 0.05:
            score += 1
            reasons.append(f'Positive 5-day momentum (+{mom_5d:.0%})')
        elif mom_5d <= -0.15:
            score -= 2
            warnings.append(f'Steep 5-day decline ({mom_5d:.0%}) — could be selling pressure')
        elif mom_5d <= -0.05:
            score -= 1
            warnings.append(f'Negative 5-day momentum ({mom_5d:.0%})')

    if market_cap:
        if market_cap < 10_000_000:
            warnings.append('Micro-cap (<$10M) — very high manipulation risk')
        elif market_cap < 50_000_000:
            warnings.append('Small market cap (<$50M) — elevated volatility risk')

    if score >= 5:
        verdict = 'STRONG BUY'
    elif score >= 3:
        verdict = 'BUY'
    elif score >= 1:
        verdict = 'WATCH'
    elif score >= -1:
        verdict = 'NEUTRAL'
    else:
        verdict = 'AVOID'

    return verdict, reasons, warnings


@app.route('/api/penny-stocks')
def api_penny_stocks():
    import yfinance as yf

    mentions, posts_by_t = _scrape_penny_reddit(limit=30)
    if not mentions:
        return jsonify([])

    # sort by mentions, take top 20 candidates
    candidates = sorted(mentions.items(), key=lambda x: -x[1])[:20]
    swing_model = load_universal('swing')

    results = []
    for ticker, mention_count in candidates:
        try:
            info = yf.Ticker(ticker).info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not price or price > 20:          # only stocks under $20
                continue
            if price < 0.001:
                continue

            mkt_cap     = info.get('marketCap')
            volume      = info.get('volume') or info.get('regularMarketVolume', 0)
            avg_volume  = info.get('averageVolume', 0)
            vol_ratio   = round(volume / avg_volume, 2) if avg_volume else None
            week52_high = info.get('fiftyTwoWeekHigh')
            week52_low  = info.get('fiftyTwoWeekLow')
            sector      = info.get('sector', 'Unknown')
            industry    = info.get('industry', '')
            name        = info.get('shortName') or info.get('longName') or ticker
            summary     = (info.get('longBusinessSummary') or '')[:300]
            pe          = info.get('trailingPE')

            # 5-day momentum
            hist    = yf.Ticker(ticker).history(period='10d', interval='1d')
            mom_5d  = None
            if len(hist) >= 5:
                mom_5d = float((hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5])

            # ML signal
            signal = None
            if swing_model:
                try:
                    df_hist = yf.Ticker(ticker).history(period='2y', interval='1d')
                    if len(df_hist) >= 250:
                        df_hist = df_hist[['Open','High','Low','Close','Volume']].copy()
                        feat_df   = add_features(df_hist)
                        feat_cols = [c for c in feat_df.columns if c not in _FEAT_DROP]
                        X = feat_df[feat_cols].replace([np.inf,-np.inf], np.nan).dropna()
                        if not X.empty:
                            signal = round(float(swing_model.predict_proba(X.iloc[[-1]])[:,1][0]), 3)
                except Exception:
                    pass

            verdict, reasons, warnings = _make_recommendation(signal, vol_ratio, mom_5d, mkt_cap)

            results.append({
                'ticker':       ticker,
                'name':         name,
                'price':        round(float(price), 4),
                'marketCap':    mkt_cap,
                'volume':       volume,
                'avgVolume':    avg_volume,
                'volRatio':     vol_ratio,
                'week52High':   round(float(week52_high), 4) if week52_high else None,
                'week52Low':    round(float(week52_low), 4) if week52_low else None,
                'sector':       sector,
                'industry':     industry,
                'pe':           round(float(pe), 1) if pe else None,
                'mom5d':        round(mom_5d * 100, 2) if mom_5d is not None else None,
                'signal':       signal,
                'mentions':     mention_count,
                'posts':        posts_by_t.get(ticker, []),
                'summary':      summary,
                'verdict':      verdict,
                'reasons':      reasons,
                'warnings':     warnings,
            })
        except Exception as e:
            print(f'[penny] {ticker}: {e}')

    results.sort(key=lambda x: (
        {'STRONG BUY': 0, 'BUY': 1, 'WATCH': 2, 'NEUTRAL': 3, 'AVOID': 4}.get(x['verdict'], 5),
        -x['mentions']
    ))
    return jsonify(results)


@app.route('/api/signal/<ticker>')
def api_single_signal(ticker):
    ticker = ticker.upper().strip()
    try:
        model = load_universal('swing')
        if model is None:
            model = load_model(ticker)
        if model is None:
            return jsonify({'error': 'Train the US swing model first (Train US Models button).'}), 404

        df = fetch(ticker, period='5y', interval='1d', use_cache=True)
        if df is None or len(df) < 250:
            return jsonify({'error': f'Not enough data for {ticker}'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        feat_df   = add_features(df)
        feat_cols = [c for c in feat_df.columns if c not in _FEAT_DROP]
        X = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty:
            return jsonify({'error': 'Could not compute features'}), 404

        def _f(v, digits=2):
            try:
                v = float(v)
                return round(v, digits) if np.isfinite(v) else None
            except Exception:
                return None

        prob      = _f(model.predict_proba(X.iloc[[-1]])[:, 1][0], 3)
        close     = df['Close'].squeeze()
        price     = _f(close.iloc[-1])
        change_1d = _f(close.pct_change().iloc[-1] * 100) or 0.0
        label     = 'BUY' if (prob or 0) >= 0.70 else ('SELL' if (prob or 1) <= 0.45 else 'HOLD')

        return jsonify({'ticker': ticker, 'signal': prob,
                        'price': price, 'change_1d': change_1d, 'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _get_news(ticker):
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
        score, headlines = 0, []
        for item in raw[:8]:
            c     = item.get('content', item) if isinstance(item, dict) else {}
            title = (c.get('title') or item.get('title', '')) if isinstance(c, dict) else ''
            if not title:
                continue
            headlines.append(title)
            tl = title.lower()
            for w in _NEWS_POS:
                if w in tl: score += 1
            for w in _NEWS_NEG:
                if w in tl: score -= 1
        return score, headlines[:4]
    except Exception:
        return 0, []


@app.route('/api/buy-now')
def api_buy_now():
    swing_model = load_universal('swing')
    all_tickers = list(dict.fromkeys(STOCKS + get_sp500()))
    candidates  = []

    for ticker in all_tickers:
        try:
            df = fetch(ticker, period='5y', interval='1d', use_cache=True)
            if df is None or len(df) < 250:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df['Close'].squeeze()
            vol   = df['Volume'].squeeze()
            price = round(float(close.iloc[-1]), 2)
            vol_r = round(float(vol.iloc[-1] / vol.iloc[-20:].mean()), 2) if len(vol) >= 20 else 1.0
            mom3d = round(float(close.pct_change(3).iloc[-1] * 100), 2)

            ml_prob = None
            if swing_model:
                feat_df   = add_features(df)
                feat_cols = [c for c in feat_df.columns if c not in _FEAT_DROP]
                X = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
                if not X.empty:
                    ml_prob = float(swing_model.predict_proba(X.iloc[[-1]])[:, 1][0])

            candidates.append({'ticker': ticker, 'price': price,
                                'vol_r': vol_r, 'mom3d': mom3d, 'ml_prob': ml_prob})
        except Exception:
            pass

    # only fetch news for top 20 by ML signal to stay fast
    candidates.sort(key=lambda x: -(x['ml_prob'] or 0))
    results = []

    for c in candidates[:20]:
        news_score, headlines = _get_news(c['ticker'])
        total, reasons = 0, []

        if c['ml_prob'] and c['ml_prob'] >= 0.72:
            total += 3
            reasons.append(f"Model confidence {round(c['ml_prob'] * 100)}%")
        elif c['ml_prob'] and c['ml_prob'] >= 0.65:
            total += 1

        if news_score >= 2:
            total += 3
            reasons.append('Strongly positive news')
        elif news_score == 1:
            total += 1
            reasons.append('Positive news coverage')
        elif news_score < 0:
            total -= 2

        if c['vol_r'] >= 2.0:
            total += 2
            reasons.append(f"Volume {c['vol_r']}x average")
        elif c['vol_r'] >= 1.4:
            total += 1
            reasons.append(f"Volume {c['vol_r']}x average")

        if c['mom3d'] >= 2.0:
            total += 1
            reasons.append(f"Up {c['mom3d']}% in 3 days")
        elif c['mom3d'] <= -3.0:
            total -= 1

        if total >= 5 and reasons:
            results.append({
                'ticker':    c['ticker'],
                'price':     c['price'],
                'signal':    round(c['ml_prob'], 3) if c['ml_prob'] else None,
                'volRatio':  c['vol_r'],
                'mom3d':     c['mom3d'],
                'newsScore': news_score,
                'headlines': headlines,
                'reasons':   reasons,
                'score':     total,
            })

    results.sort(key=lambda x: -x['score'])
    return jsonify(results[:8])


# ── Pro page (password gated) ──────────────────────────────────────────────────

PRO_PASSWORD = os.environ.get('PRO_PASSWORD', 'Marsel-Pro-2026')
_pro_tokens  = set()


def require_pro(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
        if not token or token not in _pro_tokens:
            return jsonify({'error': 'Unauthorized'}), 401
        return fn(*args, **kwargs)
    return wrapper


# Curated pro watchlist - high-liquidity quality names
PRO_DEFAULTS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
    'TSLA', 'AMZN', 'JPM', 'LLY', 'UNH',
    'V', 'COST', 'XOM', 'AVGO', 'NFLX',
    'VOLV-B.ST', 'ERIC-B.ST', 'INVE-B.ST',
]


@app.route('/pro')
def serve_pro():
    return send_file(os.path.join(os.path.dirname(__file__), 'pro.html'))


@app.route('/api/pro/login', methods=['POST'])
def pro_login():
    data = request.get_json(force=True, silent=True) or {}
    if data.get('password') == PRO_PASSWORD:
        token = secrets.token_urlsafe(24)
        _pro_tokens.add(token)
        return jsonify({'token': token, 'defaults': PRO_DEFAULTS})
    return jsonify({'error': 'Wrong password'}), 401


@app.route('/api/pro/check', methods=['GET'])
@require_pro
def pro_check():
    has_models = len(load_pro_models()) > 0
    return jsonify({'ok': True, 'trained': has_models, 'defaults': PRO_DEFAULTS})


@app.route('/api/pro/train', methods=['POST'])
@require_pro
def pro_train():
    if _task['running']:
        return jsonify({'error': 'Another job is running'}), 409

    tickers = list(dict.fromkeys(STOCKS + get_sp500()))

    def _job():
        train_pro(tickers, forward_days=3, threshold=0.015)

    threading.Thread(target=_run_task, args=(_job,), daemon=True).start()
    return jsonify({'ok': True, 'count': len(tickers)})


@app.route('/api/pro/signal/<ticker>')
@require_pro
def pro_signal(ticker):
    ticker = ticker.upper().strip()
    try:
        feature_cols = load_feature_cols()
        if not feature_cols:
            return jsonify({'error': 'Pro models not trained yet. Click Train Pro Model.'}), 503

        df = fetch(ticker, period='5y', interval='1d', use_cache=True)
        if df is None or len(df) < 300:
            return jsonify({'error': f'Not enough history for {ticker}'}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        feat = add_pro_features(df)
        # align to training feature set
        for c in feature_cols:
            if c not in feat.columns:
                feat[c] = np.nan
        X = feat[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty:
            return jsonify({'error': 'Could not compute pro features (insufficient data)'}), 404

        # Multi-timeframe: also score the weekly trend
        weekly_confirms = None
        try:
            weekly = df.resample('W').agg({'Open':'first','High':'max','Low':'min',
                                            'Close':'last','Volume':'sum'}).dropna()
            if len(weekly) >= 60:
                wfeat = add_pro_features(weekly)
                for c in feature_cols:
                    if c not in wfeat.columns:
                        wfeat[c] = np.nan
                wX = wfeat[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
                if not wX.empty:
                    wpred = predict_pro(wX.iloc[[-1]])
                    weekly_confirms = bool(wpred and wpred['ensemble'] >= 0.55)
        except Exception as e:
            print(f'[pro_signal] weekly check failed for {ticker}: {e}')

        result = predict_pro(X.iloc[[-1]])
        if not result:
            return jsonify({'error': 'No models loaded'}), 503

        prob = result['ensemble']

        # Position sizing: scale by confidence, cap at 20% of capital
        MAX_POS_PCT = 20.0
        if prob >= 0.70:
            position_size_pct = round(min(prob * MAX_POS_PCT, MAX_POS_PCT), 1)
            label = 'BUY'
        elif prob <= 0.40:
            position_size_pct = 0.0
            label = 'SELL'
        else:
            position_size_pct = 0.0
            label = 'HOLD'

        # ATR-based stops (1.5x ATR stop, 2.5x ATR target)
        close = df['Close'].squeeze()
        price = float(close.iloc[-1])
        try:
            tr = pd.concat([df['High'] - df['Low'],
                            (df['High'] - close.shift()).abs(),
                            (df['Low']  - close.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
        except Exception:
            atr = price * 0.02

        stop_loss   = round(price - 1.5 * atr, 2)
        take_profit = round(price + 2.5 * atr, 2)
        rr_ratio    = round((take_profit - price) / max(price - stop_loss, 0.01), 2)

        # Multi-timeframe filter: downgrade if weekly disagrees
        if weekly_confirms is False and label == 'BUY':
            label = 'WAIT'
            position_size_pct = round(position_size_pct / 2, 1)

        change_1d = close.pct_change().iloc[-1] * 100
        change_1d = round(float(change_1d), 2) if np.isfinite(change_1d) else 0.0

        return jsonify({
            'ticker':            ticker,
            'price':             round(price, 2),
            'change_1d':         change_1d,
            'ensemble_signal':   prob,
            'individual':        result['individual'],
            'agreement':         result['agreement'],
            'model_count':       result['model_count'],
            'weekly_confirms':   weekly_confirms,
            'label':             label,
            'position_size_pct': position_size_pct,
            'stop_loss':         stop_loss,
            'take_profit':       take_profit,
            'risk_reward':       rr_ratio,
            'atr':               round(atr, 2),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('Dashboard running at http://localhost:8080')
    print(f'Pro page: http://localhost:8080/pro  (password: {PRO_PASSWORD})')
    app.run(host='0.0.0.0', port=8080, debug=False)
