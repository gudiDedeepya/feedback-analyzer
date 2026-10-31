
# Customer Feedback Analyzer

A Streamlit app that turns raw customer reviews into insights: how customers feel, what they talk about, and what they complain about most.

**Live demo:** https://feedback-analyzer-jehavyqdjjfgh877ue3ral.streamlit.app/

## Features
- Upload a CSV of reviews (needs a `review` column; optional `date` and `rating` columns)
- Labels every review as positive, neutral or negative
- Groups reviews into topics and shows which topics have the most negative feedback
- Summarizes the main complaints per topic (Claude API if a key is set, rule-based fallback otherwise)
- Filter and browse reviews, and download the results as CSV

## Screenshots
![Dashboard](screenshots/dashboard.png)
![Topic complaints](screenshots/topics.png)
![Browse reviews](screenshots/browse.png)

## How it works
1. **Cleaning** (pandas): removes empty rows and creates a simplified text version for topic modeling.
2. **Sentiment** (VADER): scores each review from -1 to +1. I extended VADER with custom complaint words, because it missed many complaints written in neutral-sounding language.
3. **Topics** (TF-IDF + NMF): TF-IDF turns reviews into weighted word vectors and NMF splits them into topics. Each topic is named by its top keywords. Custom stop words remove filler such as "update" and "honestly".
4. **Summaries** (Claude API, optional): sends the negative reviews of each topic to a Claude model and asks for 3 bullet points, using only what is in the reviews. Without an API key, the app shows the top keywords and the most repeated complaints instead.
5. **Dashboard** (Streamlit + Plotly): charts, topic table, expandable complaint summaries, review browser, CSV download.

## Results
Tested on 300 synthetic reviews that have star ratings:

| Setup | Agreement with star rating |
|---|---|
| Plain VADER | 52% |
| VADER + custom complaint words | 70% |

I also compared 5 and 7 topics and chose 7, because the keyword lists were more coherent and mixed fewer themes.

## Limitations
- The sample data is synthetic and repetitive, so scores and topic quality will differ on real reviews.
- The custom words were chosen after looking at the same data, so the 70% is optimistic.
- VADER often labels neutral reviews as positive (only 9% of reviews came out neutral).
- It still makes mistakes, for example labelling "My refund was handled quickly with no fuss" as negative.
- Some topics still mix themes (for example app crashes with stitching problems).
- Monthly sentiment trends on the sample data are random noise, not real trends.

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
Optional: set the `ANTHROPIC_API_KEY` environment variable to enable LLM summaries.

## Tech stack
Python, pandas, scikit-learn (TF-IDF, NMF), VADER, Streamlit, Plotly, Anthropic API (optional)

## Project structure
- `make_sample_data.py`: generates the synthetic sample reviews
- `analyzer.py`: loading, cleaning, sentiment
- `topics.py`: topic modeling
- `summarizer.py`: complaint summaries (LLM or fallback)
- `app.py`: Streamlit dashboard

## Possible improvements
- Test on a real review dataset
- Replace VADER with a trained sentiment model
- Pick the number of topics automatically (for example with topic coherence)
- Merge near-duplicate complaints before summarizing

## Author
YOUR NAME
