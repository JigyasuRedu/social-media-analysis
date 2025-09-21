from typing import List
import math
import pandas as pd

def _get_transformer_pipeline():
    try:
        from transformers import pipeline
        import torch
    except Exception as e:
        raise RuntimeError("transformers or torch not installed") from e
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# Run once and reuse in a session
PIPELINE = None
def ensure_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = _get_transformer_pipeline()
    return PIPELINE

def analyze_sentiment_texts(texts: List[str], batch_size: int = 32):
    pipe = ensure_pipeline()
    results = pipe(texts, batch_size=batch_size)
    # results is list of dicts {label: 'POSITIVE', score: 0.x}
    return results

# Fallback: VADER (faster, lexicon based)
def vader_sentiment_scores(texts):
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t) for t in texts]

# Convenience: attach to dataframe
def attach_sentiment(df, text_col='clean_text'):
    df = df.copy()
    texts = df[text_col].fillna('').tolist()
    try:
        results = analyze_sentiment_texts(texts)
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
    except Exception as e:
        # fallback to vader
        scores = vader_sentiment_scores(texts)
        df['sentiment_score'] = [s['compound'] for s in scores]
        df['sentiment_label'] = df['sentiment_score'].apply(lambda s: 'POSITIVE' if s>0.05 else ('NEGATIVE' if s<-0.05 else 'NEUTRAL'))
    return df
