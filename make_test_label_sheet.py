from pathlib import Path

import pandas as pd


# Resolve paths relative to this script so it works from any directory.
project_root = Path(__file__).resolve().parent
test_path = project_root / "data" / "test.csv"
output_path = project_root / "data" / "test_label_sheet.csv"

# Load the test data.
test = pd.read_csv(test_path)

# Sample the requested rating groups with a fixed random seed.
negative = test.loc[test["rating"] <= 2].sample(n=40, random_state=7)
neutral = test.loc[test["rating"] == 3].sample(n=20, random_state=7)
positive = test.loc[test["rating"] >= 4].sample(n=40, random_state=7)

# Combine and shuffle the sampled reviews.
label_sheet = pd.concat([negative, neutral, positive], ignore_index=True)
label_sheet = label_sheet.sample(frac=1, random_state=7).reset_index(drop=True)

# Keep review identifiers and text, and leave labels empty for manual annotation.
label_sheet = label_sheet[["id", "review"]].copy()
label_sheet["my_label"] = ""

# Save the annotation sheet.
label_sheet.to_csv(output_path, index=False)

print("Please fill my_label with positive, neutral or negative.")
