import sys

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from analyzer import analyze

# Words that appear in many reviews but say nothing about the topic
EXTRA_STOP_WORDS = {
    # filler from the synthetic demo data
    "sharing", "feedback", "ordered", "weekday", "update", "honestly",
    "fair", "experience", "last", "month", "just",
    # contractions (apostrophes are removed in cleaning)
    "didnt", "dont", "doesnt", "wasnt", "isnt", "cant", "couldnt", "wouldnt",
    "wont", "im", "ive", "id", "its", "thats", "youre", "theyre",
    # generic words in clothing reviews
    "retailer", "really", "like", "got", "bought", "wear", "wore", "going",
    "way", "thought", "wanted", "tried", "think", "would", "could", "also",
    "even", "much", "still",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | EXTRA_STOP_WORDS)


def find_topics(df, n_topics=5, n_keywords=5):
    """Group reviews into topics and find the top keywords for each topic."""
    # 1. TF-IDF: turn each review into numbers. Words that are frequent in one
    #    review but rare overall get a high weight.
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.8, min_df=3)
    tfidf = vectorizer.fit_transform(df["clean"])

    # 2. NMF: split that table into "reviews x topics" and "topics x words"
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)
    weights = nmf.fit_transform(tfidf)
    words = vectorizer.get_feature_names_out()

    # 3. The top-weighted words of each topic describe what the topic is about
    topic_keywords = {}
    for i, component in enumerate(nmf.components_):
        top = component.argsort()[::-1][:n_keywords]
        topic_keywords[i] = [words[j] for j in top]

    df = df.copy()
    df["topic"] = weights.argmax(axis=1)  # each review goes to its strongest topic
    return df, topic_keywords


def topic_summary(df, topic_keywords):
    """One row per topic: keywords, size, and how negative it is."""
    rows = []
    for t, keywords in topic_keywords.items():
        sub = df[df["topic"] == t]
        share = sub["sentiment"].value_counts(normalize=True)
        rows.append({
            "topic": t,
            "keywords": ", ".join(keywords),
            "reviews": len(sub),
            "pct_negative": round(100 * share.get("negative", 0), 1),
            "avg_sentiment": round(sub["sentiment_score"].mean(), 3),
        })
    return pd.DataFrame(rows).sort_values("pct_negative", ascending=False)


def find_complaint_topics(df, n_topics=7, n_keywords=5):
    """Find topics only among negative reviews, then tag the full dataframe."""
    negative = df[df["sentiment"] == "negative"].copy()
    if len(negative) < 30:
        raise ValueError("Not enough negative reviews to find topics")

    negative, topic_keywords = find_topics(
        negative,
        n_topics=n_topics,
        n_keywords=n_keywords,
    )

    result = df.copy()
    result["topic"] = -1
    result.loc[negative.index, "topic"] = negative["topic"]
    return result, topic_keywords


def complaint_topic_summary(df, topic_keywords):
    """Summarize topics that contain only negative reviews."""
    negative = df[df["sentiment"] == "negative"]
    total_negative = len(negative)
    rows = []
    for topic, keywords in topic_keywords.items():
        sub = negative[negative["topic"] == topic]
        rows.append({
            "topic": topic,
            "keywords": ", ".join(keywords),
            "reviews": len(sub),
            "share_of_complaints": round(100 * len(sub) / total_negative, 1),
            "avg_sentiment": round(sub["sentiment_score"].mean(), 3),
        })
    return pd.DataFrame(rows).sort_values("reviews", ascending=False)


def worst_reviews(df, topic, n=3):
    """The n most negative reviews in a topic."""
    sub = df[df["topic"] == topic].sort_values("sentiment_score")
    return sub["review"].head(n).tolist()


if __name__ == "__main__":
    df = analyze("data/sample_reviews.csv")
    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    df, keywords = find_topics(df, n_topics=n_topics)
    summary = topic_summary(df, keywords)
    print(summary.to_string(index=False))
    for t in summary["topic"]:
        print(f"\nTopic {t} - most negative reviews:")
        for review in worst_reviews(df, t):
            print("  -", review)