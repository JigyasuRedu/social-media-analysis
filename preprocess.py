import re
import pandas as pd
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import emoji

STOPWORDS = set(stopwords.words('english'))

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')
RT_RE = re.compile(r'\bRT\b')
NON_ALPHANUM = re.compile(r'[^0-9a-zA-Z\s]')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub('', text)
    text = MENTION_RE.sub('', text)
    text = RT_RE.sub('', text)
    # keep hashtag words but remove '#'
    text = HASHTAG_RE.sub(lambda m: m.group(0)[1:], text)
    text = emoji.replace_emoji(text, replace='')
    text = NON_ALPHANUM.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def tokenize(text: str):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

def preprocess_dataframe(df: pd.DataFrame, text_col: str = 'content'):
    df = df.copy()
    df['clean_text'] = df[text_col].fillna('').map(clean_text)
    df['tokens'] = df['clean_text'].map(tokenize)
    df['date'] = pd.to_datetime(df['date'])
    return df
