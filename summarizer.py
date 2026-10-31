import os

import pandas as pd

MODEL = "claude-haiku-5-5"  # a small, cheap model is enough for this


def llm_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def fallback_summary(reviews, keywords):
    """No API key: show the keywords and the most repeated complaints."""
    counts = pd.Series(reviews).value_counts().head(3)
    lines = [f"Main keywords: {', '.join(keywords)}.", "Most repeated complaints:"]
    for text, n in counts.items():
        lines.append(f"- {text} (x{n})")
    return "\n".join(lines)


def llm_summary(reviews, keywords):
    """Ask Claude for a 3-bullet summary of the complaints in one topic."""
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    sample = "\n".join(f"- {r}" for r in reviews[:30])
    prompt = (
        "You are analysing customer feedback. Below are negative reviews that "
        f"belong to one topic (keywords: {', '.join(keywords)}).\n\n{sample}\n\n"
        "Write exactly 3 short bullet points summarising the main complaints. "
        "Use only what is in the reviews. Do not invent details."
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def summarize_topic(df, topic, keywords):
    """Summarize the negative reviews of one topic (LLM if possible)."""
    negatives = df[(df["topic"] == topic) & (df["sentiment"] == "negative")]["review"].tolist()
    if not negatives:
        return "No negative reviews in this topic."
    if llm_available():
        try:
            return llm_summary(negatives, keywords)
        except Exception as e:  # network, key or quota problem: do not crash the app
            return fallback_summary(negatives, keywords) + f"\n(LLM unavailable: {type(e).__name__})"
    return fallback_summary(negatives, keywords)


if __name__ == "__main__":
    import sys

    from analyzer import analyze
    from topics import find_topics, topic_summary

    n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    df = analyze("data/sample_reviews.csv")
    df, kws = find_topics(df, n_topics=n_topics)
    summary = topic_summary(df, kws)
    print("LLM mode:", "ON" if llm_available() else "OFF (using fallback)")
    for t in summary["topic"]:
        print(f"\n=== Topic {t}: {', '.join(kws[t])} ===")
        print(summarize_topic(df, t, kws[t]))