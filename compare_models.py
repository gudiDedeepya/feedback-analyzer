import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from analyzer import add_sentiment


# Keep only rows with a valid human label.
labels = pd.read_csv("data/test_label_sheet.csv")
labels = labels[labels["my_label"].isin(["positive", "neutral", "negative"])].copy()

# Load the trained review classifier.
trained_model = joblib.load("models/sentiment_lr.joblib")

# Get predictions from plain VADER, custom VADER, and the trained model.
methods = {}
methods["plain VADER"] = add_sentiment(labels, use_custom=False)["sentiment"]
methods["custom VADER"] = add_sentiment(labels, use_custom=True)["sentiment"]
methods["trained model"] = pd.Series(
    trained_model.predict(labels["review"]),
    index=labels.index,
)

label_order = ["negative", "neutral", "positive"]
summary = []

# Print the human-label evaluation for each method.
for name, predictions in methods.items():
    print(f"\n===== {name} =====")
    print(classification_report(
        labels["my_label"],
        predictions,
        labels=label_order,
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(
        labels["my_label"],
        predictions,
        labels=label_order,
    ))

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels["my_label"],
        predictions,
        labels=label_order,
        zero_division=0,
    )
    summary.append({
        "method": name,
        "negative precision": precision[0],
        "negative recall": recall[0],
        "negative F1": f1[0],
        "accuracy": accuracy_score(labels["my_label"], predictions),
    })

# Print one compact comparison table for human labels.
print("\n===== Summary =====")
print(pd.DataFrame(summary).set_index("method").round(3).to_string())

# Convert test ratings to sentiment labels and compare negative-class performance.
test = pd.read_csv("data/test.csv")
test["label"] = test["rating"].apply(
    lambda rating: "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"
)

print("\n===== Negative precision and recall against star labels =====")
for name, predictions, use_custom in [
    ("plain VADER", add_sentiment(test, use_custom=False)["sentiment"], False),
    ("custom VADER", add_sentiment(test, use_custom=True)["sentiment"], True),
    ("trained model", pd.Series(trained_model.predict(test["review"]), index=test.index), None),
]:
    precision, recall, _, _ = precision_recall_fscore_support(
        test["label"],
        predictions,
        labels=label_order,
        zero_division=0,
    )
    print(
        f"{name}: negative precision={precision[0]:.3f}, "
        f"negative recall={recall[0]:.3f}"
    )
