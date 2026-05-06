import re
import time
import requests
import pandas as pd

# S&P 100 — fallback list
SP100 = [
    'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','BRK-B','UNH','XOM',
    'JPM','JNJ','V','PG','MA','HD','CVX','MRK','ABBV','COST',
    'PEP','KO','WMT','BAC','LLY','AVGO','MCD','TMO','CSCO','ACN',
    'ABT','DHR','ADBE','NKE','TXN','PM','MS','ORCL','RTX','BMY',
    'AMD','SCHW','GS','QCOM','INTU','LOW','CAT','AXP','DE','BLK',
    'GILD','MDLZ','ADI','AMGN','C','T','SYK','ISRG','ADP','VRTX',
    'REGN','CB','MO','ZTS','CI','EOG','CME','SHW','BKNG','TJX',
    'BSX','LRCX','ETN','GD','NOC','ITW','GE','USB','NSC','WM',
    'FDX','KLAC','FCX','MCO','PNC','EMR','CL','MMM','DUK','SO',
    'NEE','ELV','HUM','OXY','AON','F','GM','UBER','ABNB','SNOW',
]

_HEADERS = {'User-Agent': 'trading-algo/1.0 (educational project)'}

# Subreddits to scrape — ordered by signal quality
_SUBS = [
    'wallstreetbets', 'stocks', 'investing', 'options',
    'StockMarket', 'Daytrading', 'pennystocks', 'smallstreetbets',
    'RobinHoodPennyStocks', 'thetagang', 'ValueInvesting',
    'dividends', 'SecurityAnalysis', 'weedstocks', 'Superstonk',
]

_BLACKLIST = {
    'A','I','OR','FOR','ARE','THE','AN','AT','IN','ON','UP','GO','IT',
    'BE','BY','TO','DO','OF','IF','US','ALL','NEW','NOW','HOW','WHY',
    'BIG','LOW','HIGH','BUY','SELL','HOLD','LOSS','GAIN','RATE','CALL',
    'PUT','GET','USD','ETF','IPO','CEO','GDP','FED','SEC','EPS','ATH',
    'DD','TA','DRS','IMO','ATM','OTM','ITM','YOY','EOD','YOLO','WSB',
    'RH','AI','EV','PE','ER','PB','ROI','APR','APY','CPI','PCE','FOMC',
    'SPY','QQQ','IWM','VIX','SPX','NDX','DJI','DJIA','SP','SMP',
    'CEO','CFO','COO','CTO','AND','BUT','NOT','WAS','HAS','HAD','ITS',
    'THIS','THAT','WITH','FROM','THEY','WILL','BEEN','HAVE','ALSO','WHEN',
    'WHAT','JUST','LIKE','GOOD','MAKE','SOME','MORE','LONG','SHORT','BOOM',
    'PUMP','DUMP','MOON','BULL','BEAR','BAGS','FOMO','BTFD','USA','UK',
    'EU','GDP','IMF','WHO','WTO','NATO','ESG','SaaS','API','ICO',
}

_TICKER_RE = re.compile(r'(?<!\w)\$?([A-Z]{2,5})(?!\w)')


def get_sp100():
    return list(SP100)


def get_sp500():
    """Fetch S&P 500 from Wikipedia. Falls back to SP100."""
    try:
        tables  = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        return [t for t in tickers if t and isinstance(t, str)]
    except Exception as e:
        print(f'[universe] S&P 500 fetch failed ({e}), using SP100 fallback')
        return list(SP100)


def get_all_listed():
    """
    Download every stock listed on NASDAQ and NYSE/AMEX from NASDAQ's
    public symbol directory. Returns ~6,000-8,000 common stock tickers.
    Falls back to S&P 500 if the download fails.
    """
    urls = [
        'https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt',
        'https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt',
    ]
    tickers = set()
    for url in urls:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            lines = r.text.splitlines()
            if not lines:
                continue
            header = lines[0].split('|')
            sym_idx = next((i for i, h in enumerate(header) if 'Symbol' in h), 0)
            for line in lines[1:]:
                parts = line.split('|')
                if len(parts) <= sym_idx:
                    continue
                sym = parts[sym_idx].strip()
                # Skip ETFs, funds, warrants, rights, units, test issues
                if not sym or not sym.isalpha() or len(sym) > 5:
                    continue
                if any(c in sym for c in ('$', '^', '.')):
                    continue
                # otherlisted has an ETF column
                if 'N' in parts and parts[-2] == 'Y':  # ETF flag
                    continue
                tickers.add(sym)
            print(f'[universe] loaded {len(tickers)} symbols so far from {url.split("/")[-1]}')
        except Exception as e:
            print(f'[universe] {url.split("/")[-1]} error: {e}')

    if len(tickers) < 100:
        print('[universe] fallback to S&P 500')
        return get_sp500()

    return sorted(tickers)


def _extract_tickers(text):
    found = []
    for m in _TICKER_RE.finditer(text):
        t = m.group(1)
        if t not in _BLACKLIST and 2 <= len(t) <= 5:
            found.append(t)
    return found


def get_reddit_tickers(limit=100):
    """
    Scrape 15 subreddits (hot + top/week feeds), extract tickers from
    post titles, bodies, and top-level comments on the 5 hottest posts
    per subreddit. Returns ranked list of (ticker, mention_count).
    """
    mentions = {}

    for sub in _SUBS:
        for feed in ('hot', 'top'):
            try:
                params = f'limit={limit}' + ('&t=week' if feed == 'top' else '')
                url = f'https://www.reddit.com/r/{sub}/{feed}.json?{params}'
                r   = requests.get(url, headers=_HEADERS, timeout=10)
                if r.status_code != 200:
                    continue
                posts = r.json()['data']['children']

                for i, post in enumerate(posts):
                    d     = post['data']
                    text  = d.get('title', '') + ' ' + d.get('selftext', '')
                    for t in _extract_tickers(text):
                        mentions[t] = mentions.get(t, 0) + 1

                    # scrape comments on top 5 posts
                    if i < 5:
                        try:
                            post_id  = d.get('id', '')
                            cr = requests.get(
                                f'https://www.reddit.com/r/{sub}/comments/{post_id}.json?limit=50',
                                headers=_HEADERS, timeout=8)
                            if cr.status_code == 200:
                                comment_tree = cr.json()
                                if len(comment_tree) > 1:
                                    for c in comment_tree[1]['data']['children']:
                                        body = c['data'].get('body', '')
                                        for t in _extract_tickers(body):
                                            mentions[t] = mentions.get(t, 0) + 1
                        except Exception:
                            pass

                print(f'[reddit] r/{sub}/{feed}: {len(posts)} posts')
                time.sleep(0.5)  # be polite to Reddit
            except Exception as e:
                print(f'[reddit] r/{sub}/{feed} error: {e}')

    ranked = sorted(mentions.items(), key=lambda x: -x[1])
    # require at least 2 mentions to filter noise
    return [(t, c) for t, c in ranked if c >= 2]
