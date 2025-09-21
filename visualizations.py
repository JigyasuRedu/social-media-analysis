import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

def plot_sentiment_distribution(df, label_col='sentiment_label', save_path=None):
    counts = df[label_col].value_counts()
    ax = counts.plot(kind='bar', title='Sentiment distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    if save_path:
        ax.figure.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_sentiment_over_time(df, date_col='date', score_col='sentiment_score', freq='D', save_path=None):
    ser = df.set_index(date_col)[score_col].resample(freq).mean().dropna()
    ax = ser.plot(title='Avg sentiment over time')
    ax.set_ylabel('Average sentiment score')
    if save_path:
        ax.figure.savefig(save_path, bbox_inches='tight')
    plt.show()

def make_wordcloud_from_texts(texts, max_words=200, save_path=None):
    joined = " ".join([t for t in texts if isinstance(t, str)])
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, max_words=max_words)
    wc.generate(joined)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
