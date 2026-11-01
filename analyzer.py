import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NEW: complaint words VADER does not treat as negative.
# VADER's word scale runs from -4 (very negative) to +4 (very positive).
CUSTOM_WORDS = {
    "late": -1.5, "delay": -1.5, "delayed": -1.5, "damaged": -2.0, "torn": -1.5,
    "expensive": -1.8, "unfair": -2.0, "ignored": -2.0, "rude": -2.5, "slow": -1.5,
    "crashing": -2.0, "crashes": -2.0, "stopped": -1.2, "nobody": -1.2,
    "transferred": -1.0, "surprise": -1.0, "apart": -1.5, "cheap": -1.5,
    "missing": -1.5,
}


def load_reviews(path):
    """Read the CSV, check it has a 'review' column, and drop empty rows."""
    df = pd.read_csv(path)
    if "review" not in df.columns:
        raise ValueError("CSV must have a 'review' column")
    df = df.dropna(subset=["review"]).copy()
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"] != ""].reset_index(drop=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def clean_for_topics(text):
    """Simplified text for topic modeling: lowercase, no links, letters only."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Two analyzers so we can compare plain VADER with the improved one
_base_analyzer = SentimentIntensityAnalyzer()
_custom_analyzer = SentimentIntensityAnalyzer()
_custom_analyzer.lexicon.update(CUSTOM_WORDS)  # NEW: add our complaint words


def label_from_score(score):
    """VADER's standard cutoffs: >= 0.05 positive, <= -0.05 negative."""
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def add_sentiment(df, use_custom=True):
    """Score each review from -1 (very negative) to +1 (very positive)."""
    df = df.copy()
    analyzer = _custom_analyzer if use_custom else _base_analyzer
    df["sentiment_score"] = df["review"].apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    df["sentiment"] = df["sentiment_score"].apply(label_from_score)
    return df


def rating_to_label(rating):
    """1-2 stars = negative, 3 = neutral, 4-5 = positive. Invalid rating -> None."""
    if pd.isna(rating) or rating < 1 or rating > 5:
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def accuracy_vs_rating(df):
    """Share of reviews where predicted sentiment matches the star rating.
    Rows with a missing or invalid rating are skipped."""
    if "rating" not in df.columns:
        return None
    ratings = pd.to_numeric(df["rating"], errors="coerce")
    true_labels = ratings.apply(rating_to_label)
    valid = true_labels.notna()
    if not valid.any():
        return None
    return (df.loc[valid, "sentiment"] == true_labels[valid]).mean()


def analyze(path, use_custom=True):
    df = load_reviews(path)
    df["clean"] = df["review"].apply(clean_for_topics)
    df = add_sentiment(df, use_custom=use_custom)
    return df


if __name__ == "__main__":
    base = analyze("data/sample_reviews.csv", use_custom=False)
    custom = analyze("data/sample_reviews.csv", use_custom=True)
    if "rating" in custom.columns:
        print("Accuracy vs rating, plain VADER :", round(accuracy_vs_rating(base), 3))
        print("Accuracy vs rating, custom words:", round(accuracy_vs_rating(custom), 3))
        print("\nRating vs predicted sentiment (custom):\n",
              pd.crosstab(custom["rating"], custom["sentiment"]))