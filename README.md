# 🍽️ Beyond the Stars: Decoding Customer Voices in Food Reviews

This project explores over 500,000 customer reviews from the **Amazon Fine Food Reviews** dataset. Using exploratory data analysis (EDA) and sentiment analysis, we uncover how customers express satisfaction and dissatisfaction beyond numeric ratings. Our goal is to bridge the gap between star ratings and the language of real feedback.

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
- Identify trends in review lengths, scores, and content
- Apply sentiment analysis (TextBlob/VADER)
- Compare linguistic sentiment scores to review scores
- Visualize relationships between text, time, and emotion

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
│   └── sentiment_analysis.py             # Sentiment scoring using TextBlob/VADER
│
├── visuals/
│   ├── wordclouds/                       # WordClouds for review texts
│   ├── histograms/                       # Score distributions, etc.
│   └── correlation_plots/                # Rating vs sentiment, etc.
│
├── results/
│   └── sentiment_scores.csv              # Saved sentiment predictions
│
├── README.md
├── requirements.txt                      # Python dependencies
└── .gitignore
```

---

## 📊 Key Analyses & Visuals

- **Score distribution bar plots**
- **Review length histogram by score**
- **Wordclouds for positive and negative reviews**
- **Frequent bigrams/trigrams across score levels**
- **Sentiment polarity histograms**
- **Correlation plots: score vs sentiment polarity**
- **Temporal review trends (optional)**

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

- NLTK, TextBlob, VADER (via nltk.sentiment.vader)

- WordCloud, Scikit-learn

- Jupyter Notebook

🔑 License
This project is licensed under the MIT License.

🙋‍♀️ Author
Created by Marwah Mahmood
[LinkedIn Profile](http://www.linkedin.com/in/MarwahFaraj) | [Email](mailto:marwah.faraj777@gmail.com)
