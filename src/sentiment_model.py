import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
import nltk
nltk.download('vader_lexicon')

def load_raw_news():
    """
    Loads the three Kaggle news datasets from the local data/ folder.
    """
    try:
        cnbc = pd.read_csv("data/cnbc_headlines.csv")
        guardian = pd.read_csv("data/guardian_headlines.csv")
        reuters = pd.read_csv("data/reuters_headlines.csv")

        df = pd.concat([cnbc, guardian, reuters], ignore_index=True)
        df = df.dropna()

        # Notebook uses Headlines + Description
        if "Headlines" in df.columns and "Description" in df.columns:
            df = df[["Headlines", "Description"]]
        else:
            raise ValueError("Expected columns 'Headlines' and 'Description' missing.")

        return df
    
    except Exception as e:
        print(f"Error loading news datasets: {e}")
        return None


def classify_with_vader(text):
    """
    Converts text → positive/neutral/negative using VADER 
    """
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
    Uses VADER-labelled descriptions to train a Linear SVC sentiment classifier
    """
    df = load_raw_news()
    if df is None:
        raise RuntimeError("Could not load news data.")

    print("Generating sentiment labels with VADER...")
    df["label"] = df["Description"].apply(classify_with_vader)

    X = df["Description"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=212
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LinearSVC())
    ])

    print("Training Linear SVC sentiment model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model for future use
    joblib.dump(pipeline, "results/sentiment_pipeline.pkl")
    print("Model saved to results/sentiment_pipeline.pkl")

    return pipeline

if __name__ == "__main__":
    train_sentiment_model()
