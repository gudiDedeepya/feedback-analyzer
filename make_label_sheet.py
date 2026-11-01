import pandas as pd

df = pd.read_csv("data/sample_reviews.csv")
sheet = df.sample(50, random_state=1)[["review"]].copy()   # no rating column on purpose
sheet["my_label"] = ""
sheet.to_csv("data/label_sheet.csv", index=True, index_label="id")
print("Saved data/label_sheet.csv. Fill my_label with positive, neutral or negative.")