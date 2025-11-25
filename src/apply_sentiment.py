import pandas as pd
import joblib

from news_data import load_and_combine_news


def label_news_with_sentiment():
    print("--- Loading combined news data ---")
    news_df = load_and_combine_news()
    print(f"Loaded {len(news_df)} news rows.")

    print("--- Loading trained sentiment pipeline ---")
    pipeline = joblib.load("results/sentiment_pipeline.pkl")

    print("--- Predicting sentiment labels ---")
    news_df["label"] = pipeline.predict(news_df["text"])

    # Map labels to numeric scores for aggregation
    label_to_score = {"negative": -1, "neutral": 0, "positive": 1}
    news_df["sentiment_score"] = news_df["label"].map(label_to_score)

    # Aggregate to daily sentiment
    daily_sent = (
        news_df.groupby("date")["sentiment_score"]
        .mean()
        .rename("daily_sentiment")
        .to_frame()
    )

    # Save for later analysis (local only, data/ is gitignored)
    daily_sent.to_csv("data/daily_sentiment_2017_2020.csv")
    print("Saved daily sentiment time series to data/daily_sentiment_2017_2020.csv")

    return daily_sent


if __name__ == "__main__":
    label_news_with_sentiment()
