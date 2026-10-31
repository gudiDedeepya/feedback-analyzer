import os
import random
from datetime import date, timedelta

import pandas as pd

random.seed(42)  # same random numbers every run, so results are repeatable

# Review templates: theme -> sentiment -> list of sentences
REVIEWS = {
    "delivery": {
        "positive": [
            "My order arrived a day early and the packaging was perfect.",
            "Super fast delivery, the courier was polite and on time.",
            "I could track my parcel the whole way and it came quickly.",
        ],
        "neutral": [
            "The delivery took about five days, which is what they said.",
            "The package arrived on the expected date.",
        ],
        "negative": [
            "My order arrived two weeks late and nobody told me why.",
            "The package was damaged and the box was torn when it reached me.",
            "The courier never showed up and tracking said delivered when it was not.",
        ],
    },
    "pricing": {
        "positive": [
            "Great prices, much cheaper than other shops.",
            "The discounts are generous and the final price had no hidden fees.",
            "Really good value for money.",
        ],
        "neutral": [
            "Prices are about the same as other stores.",
            "The price was fair, nothing special.",
        ],
        "negative": [
            "Way too expensive for what you get.",
            "They added surprise delivery charges at checkout.",
            "The price went up right before I paid and it feels unfair.",
        ],
    },
    "support": {
        "positive": [
            "Customer support answered in minutes and solved my problem.",
            "The support agent was friendly and very helpful.",
            "My refund was handled quickly with no fuss.",
        ],
        "neutral": [
            "Support replied the next day and answered my question.",
            "I contacted support once and it was an average experience.",
        ],
        "negative": [
            "Support ignored my emails for a week.",
            "The agent was rude and could not fix my issue.",
            "I was transferred three times and nobody helped me.",
        ],
    },
    "quality": {
        "positive": [
            "The product quality is excellent and feels durable.",
            "Exactly as described, well made and works perfectly.",
            "Better quality than I expected for this price.",
        ],
        "neutral": [
            "The product is okay, does the job.",
            "Quality is average, nothing to complain about.",
        ],
        "negative": [
            "It stopped working after two days, very poor quality.",
            "The item looks nothing like the photos and feels cheap.",
            "The stitching came apart on the first use.",
        ],
    },
    "website": {
        "positive": [
            "The app is easy to use and checkout is smooth.",
            "Great website, I found what I needed quickly.",
            "The search works well and the pages load fast.",
        ],
        "neutral": [
            "The website works fine, a bit plain.",
            "The app is okay, I have no strong opinion.",
        ],
        "negative": [
            "The app keeps crashing when I try to pay.",
            "The website is slow and the search never finds anything.",
            "I could not log in for two days and the reset email never came.",
        ],
    },
}

# Small extras so reviews do not look copy-pasted
OPENERS = ["", "", "Honestly, ", "To be fair, ", "Update: ", "I ordered last month. "]
CLOSERS = ["", "", "", " That is my experience.", " Just sharing my feedback.", " Ordered on a weekday."]

RATINGS = {"positive": [4, 5], "neutral": [3], "negative": [1, 2]}


def make_reviews(n=300):
    rows = []
    today = date.today()
    for _ in range(n):
        theme = random.choice(list(REVIEWS))
        # 40% positive, 20% neutral, 40% negative
        sentiment = random.choices(["positive", "neutral", "negative"], weights=[40, 20, 40])[0]
        sentence = random.choice(REVIEWS[theme][sentiment])
        text = random.choice(OPENERS) + sentence + random.choice(CLOSERS)
        rating = random.choice(RATINGS[sentiment])
        day = today - timedelta(days=random.randint(0, 180))  # last 6 months
        rows.append({"review": text, "date": day.isoformat(), "rating": rating})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = make_reviews(300)
    df.to_csv("data/sample_reviews.csv", index=False)
    print(df.head(10))
    print("\nRatings count:\n", df["rating"].value_counts().sort_index())