from pathlib import Path

import joblib
import numpy as np


# Load the trained sentiment pipeline.
project_root = Path(__file__).resolve().parent
model_path = project_root / "models" / "sentiment_lr.joblib"
pipeline = joblib.load(model_path)

# Locate the negative class coefficients and matching TF-IDF features.
classifier = pipeline.named_steps["logisticregression"]
vectorizer = pipeline.named_steps["tfidfvectorizer"]
negative_index = np.flatnonzero(classifier.classes_ == "negative")[0]
coefficients = classifier.coef_[negative_index]
feature_names = vectorizer.get_feature_names_out()

# Rank features by coefficient, with highest first and lowest last.
highest_indices = np.argsort(coefficients)[-25:][::-1]
lowest_indices = np.argsort(coefficients)[:25]

print("Highest negative coefficients:")
for index in highest_indices:
    print(f"{feature_names[index]}: {coefficients[index]:.4f}")

print("\nLowest negative coefficients:")
for index in lowest_indices:
    print(f"{feature_names[index]}: {coefficients[index]:.4f}")
