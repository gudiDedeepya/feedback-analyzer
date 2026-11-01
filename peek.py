from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

a = SentimentIntensityAnalyzer()

for word in ["great", "good", "terrible", "rude", "expensive", "late", "damaged"]:
    print(word, "->", a.lexicon.get(word, "not in VADER's dictionary"))

print("Total words in dictionary:", len(a.lexicon))