# Customer Feedback Analyzer

A Streamlit app that turns raw customer reviews into insights: how customers feel, what they complain about, and which complaint themes are biggest. Includes a held-out evaluation on real reviews, comparing a rule-based and a trained sentiment model.

**Live demo:** YOUR-STREAMLIT-APP-LINK
(The hosted demo runs on synthetic sample data with the VADER methods and rule-based summaries. The trained model and LLM summaries run locally.)

## Features
- Upload a CSV of reviews (needs a `review` column; optional `date` and `rating` columns)
- Three sentiment methods to choose from: plain VADER, VADER with custom complaint words, and a trained TF-IDF + logistic regression model (shown when the model file exists locally)
- Complaint themes: topics are found in the negative reviews only (default), or in all reviews
- Complaint summaries per topic with an LLM (Gemini free tier tested; Claude supported but untested), or a rule-based fallback when no key is set
- Charts, topic table, review browser with filters, and a CSV download

## Screenshots
![Dashboard](screenshots/dashboard.png)
![Complaint topics](screenshots/topics.png)
![Browse reviews](screenshots/browse.png)

## How it works
1. **Cleaning** (pandas): removes empty rows and creates a simplified text copy for topic modeling (lowercase, no links, apostrophes joined, letters only). VADER reads the original text, because capitals and punctuation carry feeling.
2. **Sentiment**: one of three methods:
   - **Plain VADER**: a rule-based tool with a built-in word list.
   - **VADER + custom words**: the word list extended with complaint words chosen from the dev set.
   - **Trained model**: TF-IDF (single words and word pairs) + logistic regression with balanced class weights, trained on the dev set using star-derived labels (1-2 stars = negative, 3 = neutral, 4-5 = positive).
3. **Topics** (TF-IDF + NMF): TF-IDF weights words, and NMF groups words that appear together into topics. Topics are named by their top keywords. By default topics are fitted on negative reviews only, so they describe types of complaint instead of praise.
4. **Summaries** (LLM, optional): sends each topic's negative reviews to an LLM and asks for 3 bullet points, using only what is in the reviews. Without an API key, the app shows the top keywords and the most repeated complaints.
5. **Dashboard** (Streamlit + Plotly).

## Evaluation

### Early experiment (synthetic data)
On 300 template-generated reviews, adding complaint words to VADER raised agreement with star ratings from 52% to 70%. This was optimistic: the data was repetitive and I chose the words after looking at the same data. That motivated the real-data evaluation below.

### Real data, held-out test
- **Dataset:** Women's E-Commerce Clothing Reviews (Kaggle), about 22,600 reviews after cleaning. See the dataset page for its license. The data is not included in this repository.
- **Split:** 60% dev (13,580 reviews, used for choosing custom words, training, and topic experiments) and 40% locked test set (9,054 reviews).
- **Custom words:** 14 words found on the dev set only, by comparing how often each word appears in 1-2 star vs 4-5 star reviews (for example "unflattering", "shapeless", "returned"), plus words VADER scores too weakly or as positive ("strange", "awkward", "huge", "hopes").
- **Test sample:** 100 test reviews labelled by me (50 negative, 11 neutral, 39 positive), sampled to include enough complaints. Words and models were frozen before this evaluation, and I ran it once.

| Metric (100 labelled test reviews) | Plain VADER | VADER + custom words | Trained model |
|---|---|---|---|
| Negative precision | 0.92 | 0.95 | 0.82 |
| Negative recall | 0.24 | 0.40 | 0.54 |
| Negative F1 | 0.38 | 0.56 | 0.65 |
| Accuracy | 0.50 | 0.58 | 0.61 |
| Complaints found (of 50) | 12 | 20 | 27 |

### Findings
- The trained model found the most complaints (27 of 50, vs 20 for custom VADER and 12 for plain VADER), at the cost of more false alarms (precision 0.82 vs 0.95).
- With 50 complaints in the sample, each recall value has a margin of about ±13 points. The improvement over plain VADER is clear; the gap between the trained model and custom VADER is within the margin.
- VADER's mistakes were mostly the opposite sign (30 of 50 complaints called positive). The trained model's misses were milder (13 called neutral, 10 positive).
- Real reviews are often mixed ("a beautiful dress, but the fit is strange"). A word list adds positive and negative words together, so praise often outweighs the complaint.
- **Accuracy alone is misleading.** About 77% of reviews are 4-5 stars, so predicting "positive" for everything scores about 77% against star ratings. I focus on precision and recall of the negative class.
- Inspecting the trained model's coefficients showed it rediscovered many hand-picked words ("unflattering", "returned", "cheap", "huge") and learned phrases a word list cannot ("not recommend", "didn work"), but also stylistic artifacts ("was", "were"), because it learns from star ratings.
- Neutral reviews are poorly classified by every method (only 11 examples).

### Choosing the number of topics
I score topics with NPMI coherence (do a topic's top words appear in the same reviews?) and also read the keywords.
- **All reviews:** coherence rose with the number of topics and plateaued at 12-14 (0.080-0.084). Many topics were generic praise ("love, perfect, soft"), because most reviews are positive.
- **Negative reviews only (1,421 low-star dev reviews):** coherence peaked at **7 topics** (0.071), and the topics read as complaint themes. I use 7 as the default.
- Scores are low overall (0.05-0.08), which is normal for generic review vocabulary, so I treat them as relative.

### Example output (dev set, trained model, 7 complaint topics)
Of 1,643 reviews flagged as complaints, the biggest themes were sizing and fit (19.7%), quality, material and price (19.7%), dresses with mixed praise (17.3%), fabric that looks nicer than it feels (13.8%), items that do not look like the photo or model (13.1%), boxy shirts and returns (8.9%), and sweaters and coats (7.4%). Theme names are my reading of the keywords, and the shares are estimates, because the complaints are predicted.

## Limitations
- The labelled sample is small (100 reviews, only 11 neutral), so metrics are noisy.
- The sample is balanced to include enough complaints, so precision and accuracy do not reflect the real class mix (mostly positive). Per-class recall is not affected.
- Labels come from a single annotator, and mixed reviews are subjective.
- The trained model learns from star ratings, which are a noisy proxy for sentiment. It also flags more false alarms than custom VADER.
- Complaint topics and shares depend on the sentiment step: only reviews predicted as negative are included, so about half of real complaints are missing and some false alarms are included.
- Some complaint topics mix praise and criticism ("loved the dress, but...").
- Results come from one dataset in one domain (women's clothing), so they may not transfer.
- NMF assigns each review to only one topic, even if it mentions two.
- The LLM can still invent details despite the prompt, so summaries should be checked. LLM summaries were tested on the synthetic demo data only.
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
2. Run, in order:
```
python prepare_data.py               # cleans and splits into data/dev.csv and data/test.csv
python make_test_label_sheet.py      # creates a 100-review sheet; fill the my_label column yourself
python find_missed_complaints.py     # dev set only: words more common in low-star reviews
python check_dev.py                  # dev sanity check for the custom words
python train_model.py                # trains the model on the dev set, saves models/sentiment_lr.joblib
python show_top_features.py          # which words the model learned
python compare_models.py             # final comparison on the labelled test sample
python choose_topics.py              # topic count by coherence, all reviews (dev set)
python choose_topics_negative.py     # topic count by coherence, negative reviews only (dev set)
```
The label sheet, dataset files and trained model are not included in the repository.

## Tech stack
Python, pandas, scikit-learn (TF-IDF, NMF, logistic regression, metrics), VADER, Streamlit, Plotly, Google Gemini API (free tier, optional), Anthropic API (optional)

## Project structure
- `analyzer.py`: loading, cleaning, sentiment (plain VADER, custom VADER, trained model)
- `topics.py`: topic modeling (TF-IDF + NMF), including complaint-only topics
- `summarizer.py`: complaint summaries (Gemini, Claude, or rule-based fallback)
- `app.py`: Streamlit dashboard
- `make_sample_data.py`: generates synthetic demo reviews
- `prepare_data.py`: cleans the real dataset and creates the dev/test split
- `find_missed_complaints.py`, `check_dev.py`: finding and sanity-checking custom words (dev set only)
- `train_model.py`, `show_top_features.py`: trained sentiment model and its learned words
- `make_test_label_sheet.py`, `compare_models.py`, `evaluate_labels.py`: labelled evaluation (precision, recall, F1, confusion matrices)
- `choose_topics.py`, `choose_topics_negative.py`: topic count selection with a coherence score

## Possible improvements
- Label a larger sample with more than one annotator and report agreement
- Handle mixed reviews, for example by scoring sentences separately (aspect-level sentiment)
- Compare against a fine-tuned transformer, and tune the trained model's decision threshold for the precision/recall trade-off
- Test on a second dataset or domain
- Use embedding-based topic modeling and allow multiple topics per review
- Add automated tests
- Integrate with a full-stack app (for example my Nexplay project) so the analyzer reads live reviews and ratings through an admin-only API

## Author
GUDI DEDEEPYA
