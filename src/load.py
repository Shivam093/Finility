import os
import yfinance as yf
import pandas as pd

from src.config import (
    DATA_DIR,
    YF_SP500_TICKER,
    YF_VIX_TICKER,
    START_DATE,
    END_DATE,
    SP500_CSV,
    VIX_CSV,
)


def get_market_data(start_date: str = START_DATE, end_date: str = END_DATE, save_csv: bool = True):
    """
    Fetch daily S&P 500 (^GSPC) and VIX (^VIX) data from Yahoo Finance
    between start_date and end_date.

    Returns
    -------
    sp500 : pd.DataFrame
        Historical data for S&P 500 index.
    vix : pd.DataFrame
        Historical data for VIX index.
    """
    print(f"--- Fetching market data from Yahoo Finance ({start_date} to {end_date}) ---")
    sp500 = yf.download(YF_SP500_TICKER, start=start_date, end=end_date)
    vix = yf.download(YF_VIX_TICKER, start=start_date, end=end_date)

    print(f"S&P 500 rows: {len(sp500)}, VIX rows: {len(vix)}")

    if save_csv:
        os.makedirs(DATA_DIR, exist_ok=True)

        # 🔹 Give the index a proper name so it becomes a 'Date' column in CSV
        sp500.index.name = "Date"
        vix.index.name = "Date"

        sp500.to_csv(SP500_CSV)
        vix.to_csv(VIX_CSV)

        print(f"Saved S&P 500 → {SP500_CSV}")
        print(f"Saved VIX → {VIX_CSV}")

    return sp500, vix
