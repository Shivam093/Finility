import pandas as pd

from src.config import CNBC_CSV, GUARDIAN_CSV, REUTERS_CSV, START_DATE, END_DATE


def _parse_date_from_time(time_str):
    """Extract a date from the 'Time' string."""
    if not isinstance(time_str, str):
        return None
    try:
        # Time strings look like "... , 15 May 2020"
        date_part = time_str.split(",")[-1].strip()
        return pd.to_datetime(date_part)
    except Exception:
        return None


def load_and_combine_news() -> pd.DataFrame:
    """
    Load CNBC, Guardian, and Reuters CSVs, standardize, and return a combined
    DataFrame with a 'date' column and 'text' column used for sentiment.
    """
    print("--- Loading raw news CSVs (CNBC, Guardian, Reuters) ---")
    cnbc = pd.read_csv(CNBC_CSV)
    guardian = pd.read_csv(GUARDIAN_CSV)
    reuters = pd.read_csv(REUTERS_CSV)

    cnbc["source"] = "CNBC"
    guardian["source"] = "Guardian"
    reuters["source"] = "Reuters"

    df = pd.concat([cnbc, guardian, reuters], ignore_index=True)

    # Require headlines and time
    df = df.dropna(subset=["Headlines", "Time"])

    # Parse date from "Time"
    df["date"] = df["Time"].apply(_parse_date_from_time)
    df = df.dropna(subset=["date"])

    # Restrict to 2017–2020
    df = df[
        (df["date"] >= pd.to_datetime(START_DATE)) &
        (df["date"] <= pd.to_datetime(END_DATE))
    ]

    # Build 'text' field using Description + Headlines if available
    if "Description" in df.columns:
        df["text"] = (
            df["Description"].fillna("") + " " + df["Headlines"].fillna("")
        ).str.strip()
    else:
        df["text"] = df["Headlines"].astype(str)

    df = df[df["text"].str.len() > 0]

    print(f"Combined news rows: {len(df)}")
    return df
