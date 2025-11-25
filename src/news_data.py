import os
import pandas as pd
from datetime import datetime


def _parse_date_from_time(time_str: str):
    """Extracts a date object from the 'Time' string."""
    if not isinstance(time_str, str):
        return None
    try:
        # Time strings look like "... , 15 May 2020"
        date_part = time_str.split(",")[-1].strip()
        return pd.to_datetime(date_part)
    except Exception:
        return None


def load_and_combine_news(data_dir: str = "data") -> pd.DataFrame:
    cnbc = pd.read_csv(os.path.join(data_dir, "cnbc_headlines.csv"))
    guardian = pd.read_csv(os.path.join(data_dir, "guardian_headlines.csv"))
    reuters = pd.read_csv(os.path.join(data_dir, "reuters_headlines.csv"))

    # Add source column
    cnbc["source"] = "CNBC"
    guardian["source"] = "Guardian"
    reuters["source"] = "Reuters"

    df = pd.concat([cnbc, guardian, reuters], ignore_index=True)
    df = df.dropna(subset=["Headlines", "Time"])

    # Parse date from Time
    df["date"] = df["Time"].apply(_parse_date_from_time)

    # Drop rows without valid date
    df = df.dropna(subset=["date"])

    # Optional: restrict to 2017–2020
    df = df[
        (df["date"] >= pd.to_datetime("2017-01-01")) &
        (df["date"] <= pd.to_datetime("2020-12-31"))
    ]

    # Build a text field similar to what you trained on (Description)
    # If Description is missing, fall back to Headlines
    if "Description" in df.columns:
        df["text"] = (
            df["Description"].fillna("") + " " + df["Headlines"].fillna("")
        ).str.strip()
    else:
        df["text"] = df["Headlines"].astype(str)

    df = df[df["text"].str.len() > 0]

    return df
