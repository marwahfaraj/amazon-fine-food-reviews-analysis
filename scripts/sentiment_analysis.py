from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import nltk

nltk.download('vader_lexicon')

def apply_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TextBlob and VADER sentiment scoring."""
    vader = SentimentIntensityAnalyzer()

    # TextBlob polarity
    df['textblob_polarity'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # VADER compound score
    df['vader_compound'] = df['Text'].apply(lambda x: vader.polarity_scores(x)['compound'])

    return df
