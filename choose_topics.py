import itertools

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from topics import find_topics


def npmi_coherence(docs, topic_words, eps=1e-12):
    """Average NPMI of the top words of each topic (higher = words co-occur more)."""
    vocab = sorted({w for words in topic_words.values() for w in words})
    cv = CountVectorizer(vocabulary=vocab, binary=True)
    X = cv.transform(docs).tocsc()          # documents x words, 1 if the word is present
    n_docs = X.shape[0]
    index = {w: i for i, w in enumerate(vocab)}
    doc_freq = np.asarray(X.sum(axis=0)).ravel() / n_docs

    scores = []
    for words in topic_words.values():
        pair_scores = []
        for a, b in itertools.combinations(words, 2):
            i, j = index[a], index[b]
            p_ij = X[:, i].multiply(X[:, j]).sum() / n_docs
            p_i, p_j = doc_freq[i], doc_freq[j]
            if p_i == 0 or p_j == 0:
                continue
            npmi = np.log((p_ij + eps) / (p_i * p_j)) / -np.log(p_ij + eps)
            pair_scores.append(npmi)
        if pair_scores:
            scores.append(np.mean(pair_scores))
    return float(np.mean(scores))


if __name__ == "__main__":
    from analyzer import analyze

    df = analyze("data/dev.csv")           # dev set only
    results = []
    for k in range(5, 16):
        _, kw = find_topics(df, n_topics=k, n_keywords=10)
        score = npmi_coherence(df["clean"], kw)
        results.append((k, score, kw))
        print(f"{k} topics: coherence {score:.3f}")

    best = max(results, key=lambda r: r[1])
    print(f"\nBest by coherence: {best[0]} topics")
    for t, words in best[2].items():
        print(f"  Topic {t}: {', '.join(words)}")