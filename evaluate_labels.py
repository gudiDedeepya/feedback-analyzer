import pandas as pd
from sklearn.metrics import classification_report

from analyzer import add_sentiment

sheet = pd.read_csv("data/label_sheet.csv")
sheet = sheet[sheet["my_label"].isin(["positive", "neutral", "negative"])].copy()

print("Reviews evaluated:", len(sheet))

for name, use_custom in [("plain VADER", False), ("custom words", True)]:
    scored = add_sentiment(sheet, use_custom=use_custom)
    print("\n=====", name, "=====")
    print(classification_report(
        sheet["my_label"], scored["sentiment"],
        labels=["negative", "neutral", "positive"], zero_division=0,
    ))