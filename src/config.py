"""
Configuration constants for the Finility project.

All paths, tickers, and date ranges are centralized here so that
other modules do not hard-code strings or numbers.
"""

# Base directories (these folders are local only, not tracked in git)
DATA_DIR = "data"
RESULTS_DIR = "results"

# ---------- NEWS DATA (from Kaggle, manual download, not tracked) ----------

CNBC_CSV = f"{DATA_DIR}/cnbc_headlines.csv"
GUARDIAN_CSV = f"{DATA_DIR}/guardian_headlines.csv"
REUTERS_CSV = f"{DATA_DIR}/reuters_headlines.csv"

# Daily sentiment output created by apply_sentiment.py
DAILY_SENTIMENT_CSV = f"{DATA_DIR}/daily_sentiment_2017_2020.csv"

# ---------- MARKET DATA (downloaded via yfinance) ----------

SP500_CSV = f"{DATA_DIR}/sp500_2017_2020.csv"
VIX_CSV = f"{DATA_DIR}/vix_2017_2020.csv"

# Final merged dataset
MERGED_DATA_CSV = f"{DATA_DIR}/merged_sentiment_market.csv"

# ---------- MODEL ARTIFACTS ----------

SENTIMENT_MODEL_PATH = f"{RESULTS_DIR}/sentiment_pipeline.pkl"

# ---------- TICKERS & DATE RANGE ----------

YF_SP500_TICKER = "^GSPC"
YF_VIX_TICKER = "^VIX"

START_DATE = "2017-01-01"
END_DATE = "2020-12-31"
