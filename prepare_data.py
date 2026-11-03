from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Resolve paths relative to this script so the command works from any directory.
project_root = Path(__file__).resolve().parent
raw_path = project_root / "data" / "raw" / "reviews_raw.csv"
output_dir = project_root / "data"

# Read the raw review data.
reviews = pd.read_csv(raw_path)

# Rename the source columns to the schema used by the model.
reviews = reviews.rename(columns={"Review Text": "review", "Rating": "rating"})

# Remove missing and whitespace-only reviews.
reviews = reviews.loc[
    reviews["review"].notna() & reviews["review"].astype(str).str.strip().ne("")
].copy()
reviews["review"] = reviews["review"].astype(str).str.strip()

# Keep ratings that are numeric integers in the inclusive range 1 through 5.
numeric_ratings = pd.to_numeric(reviews["rating"], errors="coerce")
valid_ratings = numeric_ratings.notna() & numeric_ratings.between(1, 5) & numeric_ratings.mod(1).eq(0)
reviews = reviews.loc[valid_ratings].copy()
reviews["rating"] = numeric_ratings.loc[valid_ratings].astype("int64")

# Remove duplicate reviews.
reviews = reviews.drop_duplicates(subset="review")

# Assign a stable row id beginning at zero.
reviews = reviews.reset_index(drop=True)
reviews["id"] = reviews.index

# Split the cleaned data into development and test sets.
dev, test = train_test_split(
    reviews,
    test_size=0.4,
    random_state=42,
    stratify=reviews["rating"],
)

# Save only the requested columns to the processed CSV files.
columns = ["id", "review", "rating"]
dev = dev[columns].sort_index()
test = test[columns].sort_index()
output_dir.mkdir(parents=True, exist_ok=True)
dev.to_csv(output_dir / "dev.csv", index=False)
test.to_csv(output_dir / "test.csv", index=False)


def print_summary(name, frame):
    """Print the row count and rating distribution for a dataset."""
    counts = frame["rating"].value_counts().sort_index()
    counts_text = ", ".join(f"{rating}: {count}" for rating, count in counts.items())
    print(f"{name}: {len(frame)} rows; rating counts: {{{counts_text}}}")


# Report dataset sizes and rating distributions.
print_summary("Full data", reviews)
print_summary("Dev", dev)
print_summary("Test", test)
