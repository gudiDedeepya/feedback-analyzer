\# Customer Feedback Analyzer\
\
A Streamlit app that turns raw customer reviews into insights: how customers feel, what they talk about, and what they complain about most.\
\
\*\*Live demo:\*\* 
(The hosted demo uses rule-based summaries. LLM summaries need an API key and are run locally.)\
\
\## Features\
\- Upload a CSV of reviews (needs a \`review\` column; optional \`date\` and \`rating\` columns)\
\- Labels every review as positive, neutral or negative\
\- Groups reviews into topics and shows which topics have the most negative feedback\
\- Summarizes the main complaints per topic with an LLM (Gemini free tier tested; Claude supported but untested), or a rule-based fallback when no key is set\
\- Filter and browse reviews, and download the results as CSV\
\
\## Screenshots\
![Dashboard]\(screenshots/dashboard.png)\
![Topic complaints]\(screenshots/topics.png)\
![Browse reviews]\(screenshots/browse.png)\
\
\## How it works\
1\. \*\*Cleaning\*\* (pandas): removes empty rows and creates a simplified text copy for topic modeling. Sentiment reads the original text, because VADER uses capitals and punctuation.\
2\. \*\*Sentiment\*\* (VADER): scores each review from -1 to +1. VADER is a rule-based tool with a built-in word list. I extended that list with custom complaint words (for example "late", "expensive", "rude"), because VADER missed many complaints written in neutral-sounding language.\
3\. \*\*Topics\*\* (TF-IDF + NMF): TF-IDF turns reviews into weighted word vectors and NMF splits them into topics. Each topic is named by its top keywords. Custom stop words remove filler such as "update" and "honestly".\
4\. \*\*Summaries\*\* (LLM, optional): sends the negative reviews of each topic to an LLM and asks for 3 bullet points, using only what is in the reviews. Without an API key, the app shows the top keywords and the most repeated complaints.\
5\. \*\*Dashboard\*\* (Streamlit + Plotly): charts, topic table, expandable complaint summaries, review browser, CSV download.\
\
\## Evaluation\
I compared plain VADER with VADER plus custom complaint words in two ways.\
\
\*\*1. Against star ratings (300 synthetic reviews).\*\* 1-2 stars = negative, 3 = neutral, 4-5 = positive. This is only a rough proxy for the true sentiment.\
\
\| Setup | Agreement with star rating |\
\|---|---|\
\| Plain VADER | 52% |\
\| VADER + custom words | 70% |\
\
\*\*2. Against a 50-review labelled sample.\*\* The labels were written by an AI assistant from the review text only (ratings hidden) and reviewed by me.\
\
\| Metric | Plain VADER | Custom words |\
\|---|---|---|\
\| Negative precision | 0.71 | 0.86 |\
\| Negative recall | 0.26 | 0.63 |\
\| Negative F1 | 0.38 | 0.73 |\
\| Neutral recall | 0.20 | 0.20 |\
\| Positive recall | 0.90 | 0.90 |\
\| Overall accuracy | 0.52 | 0.66 |\
\
\*\*Takeaway:\*\* plain VADER found only about a quarter of the real complaints. Adding complaint words roughly doubled that without lowering precision. Neutral reviews are still mostly missed, because VADER labels mildly positive wording ("okay", "fair") as positive.\
\
\*\*Topics:\*\* I compared 5 and 7 topics and chose 7, because the keyword lists were more coherent and mixed fewer themes.\
\
\*\*LLM summaries:\*\* tested with Gemini (free tier). The prompt tells the model to use only the given reviews and not to invent details.\
\
\## Limitations\
\- The data is synthetic and repetitive (about 40 base sentences), so topic quality and scores will differ on real reviews.\
\- The custom words were chosen after looking at the same dataset, so the improvement is probably overstated.\
\- The 50-review sample is small (only 10 neutral reviews), so the metrics are noisy.\
\- The sample labels were written by an AI assistant, not by independent human annotators.\
\- Star ratings are only a proxy for true sentiment.\
\- The custom lexicon ignores context: for example, "cheap" can be praise.\
\- VADER leans positive and misses most neutral reviews.\
\- Some topics still mix themes (for example app crashes with stitching problems).\
\- Text cleaning for topics removes numbers, apostrophes and non-English characters.\
\- Monthly sentiment trends on the sample data are random noise, not real trends.\
\
\## Run locally\
\`\`\`\
git clone https\://github.com/gudiDedeepya/feedback-analyzer.git\
cd feedback-analyzer\
python -m venv venv\
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate\
pip install -r requirements.txt\
python make_sample_data.py\
streamlit run app.py\
\`\`\`\
\
\*\*Optional LLM summaries:\*\* set one API key in the terminal before running the app (never commit a key).\
\`\`\`\
set GEMINI_API_KEY=your-key          # Mac/Linux: export GEMINI_API_KEY=your-key\
set ANTHROPIC_API_KEY=your-key       # alternative provider (untested)\
set GEMINI_MODEL=model-name          # optional: override the default Gemini model\
\`\`\`\
\
\*\*Reproduce the labelled evaluation:\*\*\
\`\`\`\
python make_label_sheet.py           # creates data/label_sheet.csv with 50 random reviews\
\# fill the my_label column (positive / neutral / negative), then:\
python evaluate_labels.py            # precision, recall and F1 for both versions\
\`\`\`\
\
\## Tech stack\
Python, pandas, scikit-learn (TF-IDF, NMF), VADER, Streamlit, Plotly, Google Gemini API (free tier, optional), Anthropic API (optional)\
\
\## Project structure\
\- \`make_sample_data.py\`: generates the synthetic sample reviews\
\- \`analyzer.py\`: loading, cleaning, sentiment, rating-agreement metric\
\- \`topics.py\`: topic modeling (TF-IDF + NMF)\
\- \`summarizer.py\`: complaint summaries (Gemini, Claude, or rule-based fallback)\
\- \`app.py\`: Streamlit dashboard\
\- \`make_label_sheet.py\`, \`evaluate_labels.py\`: labelled evaluation (precision, recall, F1)\
\- \`data/\`: sample reviews and the labelled sheet\
\
\## Possible improvements\
\- Test on a real, public review dataset\
\- Label a larger sample with independent human annotators\
\- Replace or complement VADER with a trained sentiment model\
\- Widen VADER's neutral threshold and evaluate it on held-out data\
\- Pick the number of topics automatically (for example with topic coherence)\
\- Merge near-duplicate complaints before summarizing\
\
\## Author\
Gudi Dedeepya