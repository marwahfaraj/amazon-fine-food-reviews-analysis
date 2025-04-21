# 🍽️ Beyond the Stars: Decoding Customer Voices in Food Reviews

This project explores over 500,000 customer reviews from the **Amazon Fine Food Reviews** dataset. Using exploratory data analysis (EDA) and sentiment analysis, we uncover how customers express satisfaction and dissatisfaction beyond numeric ratings. Our goal is to bridge the gap between star ratings and the language of real feedback.

We apply both traditional NLP techniques (TextBlob & VADER) and transformer-based sentiment models (via Hugging Face) to uncover deeper patterns in how customers write reviews.

---

## 📦 Dataset

- **Source:** [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Volume:** 568,454 reviews
- **Fields of Interest:**
  - `Text`: Full customer review
  - `Score`: Star rating (1 to 5)
  - `Summary`, `Time`, `UserId`, `ProductId`

---

## 🔍 Project Objectives

- Perform comprehensive EDA on review patterns
- Identify trends in review length, content, and star distribution
- Apply sentiment analysis using:
  - ✅ TextBlob (lexicon-based)
  - ✅ VADER (social media–tuned lexicon-based)
  - ✅ DistilBERT (transformer-based via Hugging Face)
- Compare and visualize sentiment score alignment across models
- Draw actionable insights from text-based customer feedback

---

## 📁 Project Structure

```plaintext
amazon-fine-food-nlp-eda/
│
├── data/
│   └── Reviews.csv                        # Raw dataset (downloaded from Kaggle)
│
├── notebooks/
│   └── 01_eda_sentiment_analysis.ipynb   # Main notebook for EDA + NLP
│
├── scripts/
│   ├── preprocess.py                     # Data cleaning and feature engineering
│   ├── sentiment_analysis.py             # Traditional sentiment scoring (VADER/TextBlob)
│   └── hf_sentiment_model.py             # Transformer-based sentiment scoring (DistilBERT)
│
├── visuals/
│   ├── wordclouds/                       # WordClouds for review texts
│   ├── histograms/                       # Score distributions, review lengths
│   └── correlation_plots/                # Score vs sentiment comparisons
│
├── results/
│   └── sentiment_scores.csv              # Combined sentiment results
│
├── README.md
├── requirements.txt                      # Python dependencies
└── .gitignore

---

📊 Key Analyses & Visuals
⭐ Score distribution bar plots

📏 Review length histograms

☁️ Wordclouds for high vs low star reviews

🧠 Frequent bigrams/trigrams

❤️ Sentiment polarity histograms (TextBlob & VADER)

🤖 Comparison with Hugging Face sentiment model (DistilBERT SST-2)

📉 Correlation plots: star score vs sentiment score

📅 Optional: Time series of review volume over time

---

## 🧪 How to Run

1. Clone the repo:
```bash
git clone https://github.com/yourusername/amazon-fine-food-nlp-eda.git
cd amazon-fine-food-nlp-eda
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Launch the notebook:
```bash
jupyter notebook notebooks/01_eda_sentiment_analysis.ipynb
```

🛠️ Tools & Technologies

- Python 3.9+

- Pandas, NumPy, Matplotlib, Seaborn

- NLTK, TextBlob, VADER

- Hugging Face Transformers (distilbert-base-uncased-finetuned-sst-2-english)

- WordCloud, Scikit-learn

- Jupyter Notebook

🔑 License
This project is licensed under the MIT License.

🙋‍♀️ Author
Created by Marwah Mahmood
[LinkedIn Profile](http://www.linkedin.com/in/MarwahFaraj) | [Email](mailto:marwah.faraj777@gmail.com)
