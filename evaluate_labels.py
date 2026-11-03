import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from analyzer import accuracy_vs_rating, add_sentiment, analyze

# Load only valid human labels from the test label sheet.
sheet = pd.read_csv("data/test_label_sheet.csv")
sheet = sheet[sheet["my_label"].isin(["positive", "neutral", "negative"])].copy()

print("Reviews evaluated:", len(sheet))

# Evaluate predicted sentiment against the human labels.
for name, use_custom in [("plain VADER", False), ("custom words", True)]:
    scored = add_sentiment(sheet, use_custom=use_custom)
    print("\n=====", name, "=====")
    print(classification_report(
        sheet["my_label"], scored["sentiment"],
        labels=["negative", "neutral", "positive"], zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(
        sheet["my_label"],
        scored["sentiment"],
        labels=["negative", "neutral", "positive"],
    ))

# Compare each version with the ratings in the held-out test data.
for name, use_custom in [("plain VADER", False), ("custom words", True)]:
    test = analyze("data/test.csv", use_custom=use_custom)
    accuracy = accuracy_vs_rating(test)
    print(f"\n{name} accuracy vs rating: {accuracy:.3f}")