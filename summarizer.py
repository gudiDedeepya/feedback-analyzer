import os

import pandas as pd

CLAUDE_MODEL = "claude-haiku-5-5"
# Change the Gemini model without editing code: set the GEMINI_MODEL variable
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def provider():
    """Which LLM to use: 'claude', 'gemini' or None (fallback)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return None


def llm_available():
    return provider() is not None


def fallback_summary(reviews, keywords):
    """No API key: show the keywords and the most repeated complaints."""
    counts = pd.Series(reviews).value_counts().head(3)
    lines = [f"Main keywords: {', '.join(keywords)}.", "Most repeated complaints:"]
    for text, n in counts.items():
        lines.append(f"- {text} (x{n})")
    return "\n".join(lines)


def build_prompt(reviews, keywords):
    """The same instructions are used for every LLM."""
    sample = "\n".join(f"- {r}" for r in reviews[:30])
    return (
        "You are analysing customer feedback. Below are negative reviews that "
        f"belong to one topic (keywords: {', '.join(keywords)}).\n\n{sample}\n\n"
        "Write exactly 3 short bullet points summarising the main complaints. "
        "Use only what is in the reviews. Do not invent details."
    )


def claude_summary(prompt):
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def gemini_summary(prompt):
    from google import genai

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


def llm_summary(reviews, keywords):
    prompt = build_prompt(reviews, keywords)
    if provider() == "claude":
        return claude_summary(prompt)
    return gemini_summary(prompt)


def summarize_topic(df, topic, keywords):
    """Summarize the negative reviews of one topic (LLM if possible)."""
    negatives = df[(df["topic"] == topic) & (df["sentiment"] == "negative")]["review"].tolist()
    if not negatives:
        return "No negative reviews in this topic."
    if llm_available():
        try:
            return llm_summary(negatives, keywords)
        except Exception as e:  # network, key, quota or model problem: do not crash the app
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
    print("LLM provider:", provider() or "none (using fallback)")
    for t in summary["topic"]:
        print(f"\n=== Topic {t}: {', '.join(kws[t])} ===")
        print(summarize_topic(df, t, kws[t]))