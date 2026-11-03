from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Load the development data only.
project_root = Path(__file__).resolve().parent
dev = pd.read_csv(project_root / "data" / "dev.csv")

# Split development reviews by extreme ratings.
low = dev.loc[dev["rating"].le(2)]
high = dev.loc[dev["rating"].ge(4)]
n_low = len(low)
n_high = len(high)

# Fit one binary word vocabulary on all development reviews.
vectorizer = CountVectorizer(
    stop_words="english",
    binary=True,
    min_df=20,
)
word_matrix = vectorizer.fit_transform(dev["review"].fillna("").astype(str))
low_matrix = vectorizer.transform(low["review"].fillna("").astype(str))
high_matrix = vectorizer.transform(high["review"].fillna("").astype(str))

# Count the number of reviews in each group containing each word.
feature_names = vectorizer.get_feature_names_out()
low_counts = low_matrix.sum(axis=0).A1
high_counts = high_matrix.sum(axis=0).A1

# Apply the smoothed low-to-high frequency ratio.
rows = []
for word, low_count, high_count in zip(feature_names, low_counts, high_counts):
    if low_count >= 25:
        ratio = (
            (low_count + 1) / (n_low + 2)
        ) / (
            (high_count + 1) / (n_high + 2)
        )
        rows.append((word, int(low_count), int(high_count), ratio))

# Print the highest-ratio words and group sizes.
print(f"n_low: {n_low}")
print(f"n_high: {n_high}")
print("\nword, low_count, high_count, ratio")
for word, low_count, high_count, ratio in sorted(rows, key=lambda row: row[3], reverse=True)[:80]:
    print(f"{word}, {low_count}, {high_count}, {ratio:.1f}")
