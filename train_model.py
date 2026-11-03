from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# Load the development split only.
project_root = Path(__file__).resolve().parent
dev = pd.read_csv(project_root / "data" / "dev.csv")

# Convert star ratings into sentiment labels.
dev["label"] = dev["rating"].apply(
    lambda rating: (
        "negative" if rating <= 2
        else "neutral" if rating == 3
        else "positive" if rating >= 4
        else None
    )
)
dev = dev.loc[dev["label"].notna()].copy()

# Build the word and bigram TF-IDF logistic regression pipeline.
pipeline = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True,
    ),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)

# Sanity-check the model on a stratified holdout from dev.
train_data, validation_data = train_test_split(
    dev,
    test_size=0.2,
    random_state=42,
    stratify=dev["label"],
)
pipeline.fit(train_data["review"], train_data["label"])
validation_predictions = pipeline.predict(validation_data["review"])
print(classification_report(
    validation_data["label"],
    validation_predictions,
    labels=["negative", "neutral", "positive"],
    zero_division=0,
))

# Refit on all development reviews and save the final pipeline.
pipeline.fit(dev["review"], dev["label"])
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)
joblib.dump(pipeline, models_dir / "sentiment_lr.joblib")
print(f"Saved model to {models_dir / 'sentiment_lr.joblib'}")
