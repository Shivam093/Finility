import os
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.news_data import load_and_combine_news
from src.config import RESULTS_DIR, SENTIMENT_MODEL_PATH


def _ensure_nltk_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def _vader_label(text: str) -> str:
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def train_sentiment_model():
    """
    Train a Linear SVC sentiment classifier using VADER-generated labels
    on combined news text, and save it to SENTIMENT_MODEL_PATH.
    """
    print("--- Training sentiment model (Linear SVC) ---")
    _ensure_nltk_vader()

    df = load_and_combine_news()
    print("Generating VADER labels...")
    df["label"] = df["text"].apply(_vader_label)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=212
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LinearSVC())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(pipeline, SENTIMENT_MODEL_PATH)
    print(f"Model saved to {SENTIMENT_MODEL_PATH}")

    return pipeline


if __name__ == "__main__":
    train_sentiment_model()
