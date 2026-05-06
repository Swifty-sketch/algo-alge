import re
import requests

# S&P 100 — top US stocks by market cap
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

_HEADERS  = {'User-Agent': 'trading-algo/1.0 (educational project)'}
_SUBS     = ['wallstreetbets', 'stocks', 'investing', 'options']
_BLACKLIST = {
    'A','I','OR','FOR','ARE','THE','AN','AT','IN','ON','UP','GO','IT',
    'BE','BY','TO','DO','OF','IF','US','ALL','NEW','NOW','HOW','WHY',
    'BIG','LOW','HIGH','BUY','SELL','HOLD','LOSS','GAIN','RATE','CALL',
    'PUT','GET','USD','ETF','IPO','CEO','GDP','FED','SEC','EPS','ATH',
    'DD','TA','DRS','IMO','ATM','OTM','ITM','YOY','EOD','YOLO','WSB',
    'RH','AI','EV','PE','ER','PB','ROI','APR','APY','CPI','PCE','FOMC',
    'SPY','QQQ','IWM','VIX','SPX','NDX','DJI','DJIA','SP','SMP',
}
_TICKER_RE = re.compile(r'(?<!\w)\$?([A-Z]{2,5})(?!\w)')


def get_sp100():
    return list(SP100)


def get_reddit_tickers(limit=30):
    mentions = {}
    for sub in _SUBS:
        try:
            url = f'https://www.reddit.com/r/{sub}/hot.json?limit={limit}'
            r   = requests.get(url, headers=_HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            posts = r.json()['data']['children']
            for post in posts:
                title = post['data'].get('title', '')
                body  = post['data'].get('selftext', '')
                for m in _TICKER_RE.finditer(title + ' ' + body):
                    t = m.group(1)
                    if t not in _BLACKLIST and 2 <= len(t) <= 5:
                        mentions[t] = mentions.get(t, 0) + 1
            print(f'[reddit] r/{sub}: {len(posts)} posts scanned')
        except Exception as e:
            print(f'[reddit] r/{sub} error: {e}')

    ranked = sorted(mentions.items(), key=lambda x: -x[1])
    return [(t, c) for t, c in ranked if c >= 2][:25]
