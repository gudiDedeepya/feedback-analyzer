# Customer Feedback Analyzer

A Streamlit app that turns raw customer reviews into insights: how customers feel, what they talk about, and what they complain about most. Includes an evaluation of the sentiment step on real reviews, with a held-out test set.

**Live demo:** YOUR-STREAMLIT-APP-LINK
(The hosted demo runs on synthetic sample data and uses rule-based summaries. LLM summaries need an API key and are run locally.)

## Features
- Upload a CSV of reviews (needs a `review` column; optional `date` and `rating` columns)
- Labels every review as positive, neutral or negative
- Groups reviews into topics and shows which topics have the most negative feedback
- Summarizes the main complaints per topic with an LLM (Gemini free tier tested; Claude supported but untested), or a rule-based fallback when no key is set
- Filter and browse reviews, and download the results as CSV

## Screenshots
![Dashboard](screenshots/dashboard.png)
![Topic complaints](screenshots/topics.png)
![Browse reviews](screenshots/browse.png)

## How it works
1. **Cleaning** (pandas): removes empty rows and creates a simplified text copy for topic modeling. Sentiment reads the original text, because VADER uses capitals and punctuation.
2. **Sentiment** (VADER): scores each review from -1 to +1. VADER is a rule-based tool with a built-in word list. I extended that list with custom complaint words (see Evaluation).
3. **Topics** (TF-IDF + NMF): TF-IDF turns reviews into weighted word vectors and NMF splits them into topics. Each topic is named by its top keywords. Custom stop words remove filler words.
4. **Summaries** (LLM, optional): sends the negative reviews of each topic to an LLM and asks for 3 bullet points, using only what is in the reviews. Without an API key, the app shows the top keywords and the most repeated complaints.
5. **Dashboard** (Streamlit + Plotly): charts, topic table, expandable complaint summaries, review browser, CSV download.

## Evaluation

### Early experiment (synthetic data)
On 300 template-generated reviews, adding complaint words to VADER raised agreement with star ratings from 52% to 70%. This was optimistic: the data was repetitive and I chose the words after looking at the same data. That motivated the real-data evaluation below.

### Real data, held-out test
- **Dataset:** Women's E-Commerce Clothing Reviews (Kaggle), about 22,600 reviews after cleaning. See the dataset page for its license. The data is not included in this repository.
- **Split:** 60% dev (13,580 reviews, used to choose custom words) and 40% locked test set (9,054 reviews).
- **Custom words:** 14 words (for example "unflattering", "shapeless", "returned") found on the dev set only, by comparing how often each word appears in 1-2 star vs 4-5 star reviews. I also raised or flipped words VADER scores too weakly or as positive ("strange", "awkward", "huge", "hopes").
- **Dev sanity check (against star ratings):** 1-2 star reviews flagged negative rose from 26.9% to 38.1%, while 4-5 star reviews flagged negative stayed low (1.9% to 2.3%).
- **Test sample:** 100 test reviews labelled by me (50 negative, 11 neutral, 39 positive), sampled to include enough complaints. The words were frozen before I ran this evaluation, and I ran it once.

| Metric (100 labelled test reviews) | Plain VADER | VADER + custom words |
|---|---|---|
| Negative precision | 0.92 | 0.95 |
| Negative recall | 0.24 | 0.40 |
| Negative F1 | 0.38 | 0.56 |
| Positive recall | 0.97 | 0.97 |
| Neutral recall | 0.00 | 0.00 |
| Accuracy | 0.50 | 0.58 |

### Findings
- The custom words found 8 more of the 50 real complaints (12 to 20) with almost no extra false alarms, a real but modest gain.
- **Most complaints are still missed (60%).** Real reviews mix praise and criticism ("a beautiful dress, but the fit is strange"), and a word-list approach adds the positive and negative words together, so the praise often wins.
- **Accuracy alone is misleading.** About 77% of reviews are 4-5 stars, so predicting "positive" for everything scores about 77% against star ratings. VADER scored 0.779 and the custom version 0.787 on the full test set, barely above that trivial baseline, so I use precision and recall on the negative class instead.
- VADER never predicted "neutral" on these long reviews.

## Limitations
- The labelled sample is small (100 reviews, only 11 neutral), so the metrics are noisy.
- The sample is balanced to include enough complaints, so precision and accuracy do not reflect the real class mix (mostly positive reviews). Per-class recall is not affected by this.
- Labels come from a single annotator (me), and mixed reviews are subjective.
- Star ratings are only a proxy for true sentiment and are used only for the dev check and the full-test comparison.
- A dictionary approach cannot handle mixed reviews or context ("cheap" can be praise).
- The dashboard's demo data is synthetic and repetitive, and topic quality on the real dataset has not been tuned yet.
- Text cleaning for topics removes numbers and non-English characters, and splits words at apostrophes.
- Each review is assigned to only one topic, even if it mentions two.
- The LLM can still invent details despite the prompt, so summaries should be checked against the source reviews.
- Reviews are sent to a third-party API when LLM summaries are on, so private data should not be used.

## Run locally
```
git clone https://github.com/gudiDedeepya/feedback-analyzer.git
cd feedback-analyzer
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python make_sample_data.py
streamlit run app.py
```

**Optional LLM summaries:** set one API key in the terminal before running the app (never commit a key).
```
set GEMINI_API_KEY=your-key          # Mac/Linux: export GEMINI_API_KEY=your-key
set ANTHROPIC_API_KEY=your-key       # alternative provider (untested)
set GEMINI_MODEL=model-name          # optional: override the default Gemini model
```

**Reproduce the real-data evaluation:**
1. Download the Women's E-Commerce Clothing Reviews dataset from Kaggle and save it as `data/raw/reviews_raw.csv`.
2. Run the steps in order:
```
python prepare_data.py               # cleans and splits into data/dev.csv and data/test.csv
python make_test_label_sheet.py      # creates a 100-review sheet; fill the my_label column yourself
python find_missed_complaints.py     # dev set only: words more common in low-star reviews
python check_dev.py                  # dev sanity check
python evaluate_labels.py            # final evaluation on the labelled test sample
```
The label sheet is not included, because it contains text from the dataset.

## Tech stack
Python, pandas, scikit-learn (TF-IDF, NMF, metrics), VADER, Streamlit, Plotly, Google Gemini API (free tier, optional), Anthropic API (optional)

## Project structure
- `make_sample_data.py`: generates synthetic demo reviews
- `analyzer.py`: loading, cleaning, sentiment (plain and custom VADER), rating-agreement metric
- `topics.py`: topic modeling (TF-IDF + NMF)
- `summarizer.py`: complaint summaries (Gemini, Claude, or rule-based fallback)
- `app.py`: Streamlit dashboard
- `prepare_data.py`: cleans the real dataset and creates the dev/test split
- `make_test_label_sheet.py`, `evaluate_labels.py`: labelled evaluation (precision, recall, F1, confusion matrix)
- `find_missed_complaints.py`, `check_dev.py`: finding and sanity-checking custom words on the dev set

## Possible improvements
- Train a supervised model (TF-IDF + logistic regression, or a fine-tuned transformer) and compare it against VADER on the same labelled test sample
- Handle mixed reviews, for example by scoring sentences separately or allowing multiple topics per review
- Label a larger sample with more than one annotator and report agreement
- Pick the number of topics with a coherence score, and tune topics on the real dataset
- Fix apostrophe handling in the topic-cleaning step
- Integrate with a full-stack app (for example my Nexplay project) so the analyzer reads live reviews and ratings through an admin-only API

## Author
GUDI DEDEEPYA