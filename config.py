STOCKS = [
    # Large cap
    'VOLV-B.ST', 'ERIC-B.ST', 'INVE-B.ST', 'SAND.ST',
    'SEB-A.ST',  'SWED-A.ST', 'HM-B.ST',   'AZN.ST',
    'ASSA-B.ST', 'ATCO-A.ST',
    # Banks & finance
    'NDA-SE.ST', 'KINV-B.ST', 'LATO-B.ST',
    # Industrials
    'ABB.ST',    'ALFA.ST',   'ALIV-SDB.ST', 'HEXA-B.ST',
    'HUSQ-B.ST', 'SKA-B.ST',  'SKFB.ST',     'SSAB-A.ST',
    'SWEC-B.ST', 'TREL-B.ST', 'SAAB-B.ST',
    # Consumer & retail
    'ICA.ST',    'ELUX-B.ST', 'BOL.ST',
    # Telecom & tech
    'TEL2-B.ST', 'TELIA.ST',  'SINCH.ST',
    # Healthcare
    'EKTA-B.ST', 'GETI-B.ST', 'NIBE-B.ST',
    # Real estate
    'FABG.ST',   'PEAB-B.ST', 'CAST.ST',
]

PERIOD   = '5y'
INTERVAL = '1d'

FORWARD_DAYS = 5
THRESHOLD    = 0.02    # 2% gain = bullish label
TEST_SPLIT   = 0.2

XGB_PARAMS = {
    'n_estimators':     300,
    'max_depth':        4,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'eval_metric':      'logloss',
    'random_state':     42,
}

INITIAL_CASH   = 100_000    # SEK
COMMISSION     = 0.002      # 0.2% per trade (Avanza/Nordnet typical)
ENTER_THRESH   = 0.70       # ML probability to go long
EXIT_THRESH    = 0.45       # ML probability to exit
