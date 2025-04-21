import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def predict_hf_sentiment(text: str) -> float:
    """Predict sentiment using Hugging Face DistilBERT model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits[0].numpy()
    probs = softmax(scores)
    return float(probs[1])  # Probability of "positive"

def apply_hf_sentiment(df: pd.DataFrame, text_col: str = 'Text', sample_size: int = None) -> pd.DataFrame:
    """Apply Hugging Face sentiment model to a DataFrame."""
    if sample_size:
        df = df.sample(sample_size, random_state=42).copy()

    df['hf_sentiment'] = df[text_col].apply(predict_hf_sentiment)
    return df
