import pandas as pd

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess Amazon food reviews dataset."""
    df = pd.read_csv(file_path)

    # Drop rows with null reviews or scores
    df = df.dropna(subset=['Text', 'Score'])

    # Optional: Filter out neutral reviews if needed (Score == 3)
    df = df[df['Score'] != 3]

    # Create binary target: positive (4–5), negative (1–2)
    df['label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

    # Create length-based features
    df['review_length'] = df['Text'].apply(lambda x: len(str(x)))
    df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))

    return df
