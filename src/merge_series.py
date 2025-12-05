import pandas as pd

from src.config import (
    SP500_CSV,
    VIX_CSV,
    DAILY_SENTIMENT_CSV,
    MERGED_DATA_CSV,
)


def _load_sp500() -> pd.DataFrame:
    """
    Load S&P 500 CSV and return a DataFrame indexed by Date with a 'return' column.
    Robust to minor header/format issues.
    """
    print(f"--- Loading S&P 500 data from {SP500_CSV} ---")
    df = pd.read_csv(SP500_CSV)

    # Prefer a 'Date' column if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")
    else:
        # Fallback: treat the first column as Date
        first_col = df.columns[0]
        print(f"No 'Date' column found, treating '{first_col}' as Date.")
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.dropna(subset=[first_col])
        df = df.set_index(first_col)
        df.index.name = "Date"

    # Make sure Close is numeric, drop any junk rows
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # Compute daily returns
    df["return"] = df["Close"].pct_change()
    return df


def _load_vix() -> pd.DataFrame:
    """
    Load VIX CSV and return a DataFrame indexed by Date with a 'vix' column.
    Robust to minor header/format issues.
    """
    print(f"--- Loading VIX data from {VIX_CSV} ---")
    df = pd.read_csv(VIX_CSV)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")
    else:
        first_col = df.columns[0]
        print(f"No 'Date' column found, treating '{first_col}' as Date.")
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.dropna(subset=[first_col])
        df = df.set_index(first_col)
        df.index.name = "Date"

    # Close becomes 'vix'
    df["vix"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["vix"])
    return df


def merge_datasets():
    """
    Merge S&P 500 returns, daily sentiment, and VIX into a single DataFrame
    and save it to MERGED_DATA_CSV.
    """
    print("--- Loading daily sentiment ---")
    daily_sent = pd.read_csv(DAILY_SENTIMENT_CSV)
    daily_sent["date"] = pd.to_datetime(daily_sent["date"], format="%Y-%m-%d")
    daily_sent = daily_sent.set_index("date")

    sp500 = _load_sp500()
    vix = _load_vix()

    print("--- Merging S&P returns with sentiment ---")
    merged = sp500[["return"]].merge(
        daily_sent, left_index=True, right_index=True, how="left"
    )

    print("--- Merging with VIX ---")
    merged = merged.merge(
        vix[["vix"]], left_index=True, right_index=True, how="left"
    )

    print(f"Saving merged dataset → {MERGED_DATA_CSV}")
    merged.to_csv(MERGED_DATA_CSV)

    return merged


if __name__ == "__main__":
    df = merge_datasets()
    print(df.head())
