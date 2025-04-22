import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
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

def apply_hf_sentiment(df: pd.DataFrame, text_column="Text") -> pd.DataFrame:
    """
    Applies Hugging Face DistilBERT sentiment pipeline on full dataset using batching.
    Returns a copy of the DataFrame with an added column 'hf_sentiment' (positivity probability).
    """

    # Load sentiment pipeline
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Create output list
    sentiments = []

    # Apply in batches (tqdm for progress bar)
    for result in tqdm(classifier(df[text_column].tolist(), truncation=True, batch_size=32), desc="Applying DistilBERT"):
        # Convert 'LABEL_1' (Positive) to its probability
        score = result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]
        sentiments.append(score)

    # Add to DataFrame
    df_copy = df.copy()
    df_copy["hf_sentiment"] = sentiments
    return df_copy
