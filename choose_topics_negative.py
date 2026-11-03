from analyzer import analyze
from choose_topics import npmi_coherence
from topics import find_topics

df = analyze("data/dev.csv")
neg = df[df["rating"] <= 2].reset_index(drop=True)   # dev set only
print("Negative reviews:", len(neg))

results = []
for k in range(3, 11):
    _, kw = find_topics(neg, n_topics=k, n_keywords=10)
    score = npmi_coherence(neg["clean"], kw)
    results.append((k, score, kw))
    print(f"{k} topics: coherence {score:.3f}")

best = max(results, key=lambda r: r[1])
print(f"\nBest by coherence: {best[0]} topics")
for t, words in best[2].items():
    print(f"  Topic {t}: {', '.join(words)}")