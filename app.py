import io

import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import accuracy_vs_rating, analyze
from summarizer import llm_available, summarize_topic
from topics import find_topics, topic_summary

SAMPLE_PATH = "data/sample_reviews.csv"
COLORS = {"positive": "#2e7d32", "neutral": "#9e9e9e", "negative": "#c62828"}

st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide")
st.title("Customer Feedback Analyzer")
st.caption("Upload customer reviews to see sentiment, topics and the main complaints.")


@st.cache_data
def run_pipeline(file_bytes, n_topics, use_custom):
    """Run the whole analysis. Cached, so it only re-runs when inputs change."""
    source = io.BytesIO(file_bytes) if file_bytes else SAMPLE_PATH
    df = analyze(source, use_custom=use_custom)
    df, keywords = find_topics(df, n_topics=n_topics)
    summary = topic_summary(df, keywords)
    return df, keywords, summary


@st.cache_data(show_spinner="Summarizing complaints...")
def cached_summary(df, topic, keywords):
    """Cached so the LLM is not called again on every click."""
    return summarize_topic(df, topic, list(keywords))


# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload a CSV (needs a 'review' column)", type="csv")
    n_topics = st.slider("Number of topics", 3, 10, 7)
    use_custom = st.checkbox("Use custom complaint words", value=True)
    st.write("LLM summaries:", "ON" if llm_available() else "OFF (rule-based fallback)")

# ---------- Run the analysis ----------
file_bytes = uploaded.getvalue() if uploaded else None
try:
    df, keywords, summary = run_pipeline(file_bytes, n_topics, use_custom)
except Exception as e:
    st.error(f"Could not analyze this file: {e}")
    st.stop()

if not uploaded:
    st.info("Showing the built-in sample data (synthetic reviews). Upload your own CSV in the sidebar.")

labels = {t: f"Topic {t}: {', '.join(kw[:3])}" for t, kw in keywords.items()}

# ---------- Headline numbers ----------
share = df["sentiment"].value_counts(normalize=True) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", len(df))
c2.metric("Positive", f"{share.get('positive', 0):.0f}%")
c3.metric("Neutral", f"{share.get('neutral', 0):.0f}%")
c4.metric("Negative", f"{share.get('negative', 0):.0f}%")

if "rating" in df.columns:
    try:
        st.caption(f"Predicted sentiment agrees with the star rating for {accuracy_vs_rating(df):.0%} of reviews.")
    except Exception:
        pass

# ---------- Charts ----------
left, right = st.columns(2)

with left:
    counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
    counts = counts.reset_index()
    counts.columns = ["sentiment", "reviews"]
    fig1 = px.bar(counts, x="sentiment", y="reviews", color="sentiment",
                  color_discrete_map=COLORS, title="Sentiment of reviews")
    st.plotly_chart(fig1)

with right:
    chart_df = summary.assign(label=summary["topic"].map(labels)).sort_values("pct_negative")
    fig2 = px.bar(chart_df, x="pct_negative", y="label", orientation="h",
                  title="Share of negative reviews by topic (%)",
                  labels={"pct_negative": "% negative", "label": "Topic"})
    st.plotly_chart(fig2)

if "date" in df.columns and df["date"].notna().any():
    trend = (
        df.dropna(subset=["date"])
        .assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["sentiment_score"].mean()
    )
    fig3 = px.line(trend, x="month", y="sentiment_score", markers=True,
                   title="Average sentiment by month",
                   labels={"sentiment_score": "Average sentiment (-1 to +1)", "month": "Month"})
    st.plotly_chart(fig3)

# ---------- Topic table and complaint summaries ----------
st.subheader("Topics")
table = summary.assign(topic_name=summary["topic"].map(labels))
st.dataframe(table[["topic_name", "keywords", "reviews", "pct_negative", "avg_sentiment"]],
             hide_index=True)

st.subheader("Main complaints by topic")
for _, row in summary.iterrows():
    t = row["topic"]
    with st.expander(f"{labels[t]} - {row['pct_negative']}% negative"):
        st.write(cached_summary(df, t, tuple(keywords[t])))

# ---------- Browse reviews ----------
st.subheader("Browse reviews")
f1, f2 = st.columns(2)
sent_filter = f1.multiselect("Sentiment", ["positive", "neutral", "negative"], default=["negative"])
all_topics = sorted(df["topic"].unique())
topic_filter = f2.multiselect("Topic", all_topics, default=all_topics)
view = df[df["sentiment"].isin(sent_filter) & df["topic"].isin(topic_filter)]
show_cols = [c for c in ["review", "sentiment", "sentiment_score", "topic", "rating", "date"] if c in view.columns]
st.dataframe(view[show_cols], hide_index=True)

# ---------- Download ----------
st.download_button(
    "Download results (CSV)",
    df.drop(columns=["clean"]).to_csv(index=False).encode("utf-8"),
    file_name="feedback_results.csv",
    mime="text/csv",
)