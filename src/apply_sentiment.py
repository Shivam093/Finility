import os
import joblib
import pandas as pd

from src.news_data import load_and_combine_news
from src.config import DAILY_SENTIMENT_CSV, SENTIMENT_MODEL_PATH, DATA_DIR, START_DATE, END_DATE


def label_news_with_sentiment():
    """
    Load combined news, apply trained sentiment model,
    aggregate into a daily sentiment index, and save to DAILY_SENTIMENT_CSV.
    """
    print("--- Loading combined news data ---")
    news_df = load_and_combine_news()

    if not os.path.exists(SENTIMENT_MODEL_PATH):
        raise RuntimeError(
            f"Sentiment model not found at {SENTIMENT_MODEL_PATH}. "
            "Run train_sentiment_model() first."
        )

    print(f"--- Loading sentiment model from {SENTIMENT_MODEL_PATH} ---")
    model = joblib.load(SENTIMENT_MODEL_PATH)

    print("Predicting sentiment labels for all headlines...")
    news_df["pred_label"] = model.predict(news_df["text"])

    # Map labels to numeric scores: positive=1, neutral=0, negative=-1
    score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    news_df["sentiment_score"] = news_df["pred_label"].map(score_map).fillna(0.0)

    print("Aggregating to daily sentiment index...")
    daily = (
        news_df
        .groupby("date")["sentiment_score"]
        .mean()
        .rename("daily_sentiment")
        .reset_index()
    )

    # Ensure date range
    daily = daily[
        (daily["date"] >= pd.to_datetime(START_DATE)) &
        (daily["date"] <= pd.to_datetime(END_DATE))
    ]

    os.makedirs(DATA_DIR, exist_ok=True)
    daily.to_csv(DAILY_SENTIMENT_CSV, index=False)
    print(f"Saved daily sentiment → {DAILY_SENTIMENT_CSV}")

    return daily


if __name__ == "__main__":
    label_news_with_sentiment()
