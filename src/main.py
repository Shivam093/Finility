"""
main.py – pipeline runner for Finility project.

This script:
1) Downloads market data (S&P 500 + VIX) via yfinance
2) Trains the sentiment model
3) Applies the sentiment model to headlines to create daily sentiment
4) Merges sentiment with S&P 500 returns and VIX into a final dataset
"""

from src.load import get_market_data
from src.sentiment_model import train_sentiment_model
from src.apply_sentiment import label_news_with_sentiment
from src.merge_series import merge_datasets


def main():
    # 1) Download market data and save to CSVs
    get_market_data()

    # 2) Train sentiment classification model and save to results/sentiment_pipeline.pkl
    train_sentiment_model()

    # 3) Apply model to all headlines and create daily_sentiment_2017_2020.csv
    label_news_with_sentiment()

    # 4) Merge S&P 500 returns, sentiment, and VIX into merged_sentiment_market.csv
    merge_datasets()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
