import os
import pandas as pd
from data_collect import fetch_with_snscrape
from preprocess import preprocess_dataframe
from sentiment_analysis import attach_sentiment
from topic_modeling import build_lda_model, get_topics_as_list
from visualizations import plot_sentiment_distribution, plot_sentiment_over_time, make_wordcloud_from_texts

def main():
    # Example: search for tweets about "iPhone"
    query = "iPhone lang:en since:2025-09-01 until:2025-09-21"
    print("Collecting tweets...")
    df = fetch_with_snscrape(query, limit=2000)
    print(f"Collected {len(df)} items")

    print("Preprocessing...")
    df = preprocess_dataframe(df)

    print("Sentiment analysis...")
    df = attach_sentiment(df)

    print("Topic modeling (this may take a while)...")
    tokens = df['tokens'].tolist()
    # filter docs with tokens
    tokens = [t for t in tokens if len(t) > 0]
    if tokens:
        lda, corpus, dictionary = build_lda_model(tokens, num_topics=5, passes=10)
        topics = get_topics_as_list(lda, dictionary, topn=8)
        print("Identified topics:")
        for i, t in enumerate(topics):
            print(i, ", ".join(t[:8]))
    else:
        print("Not enough tokenized documents for topic modeling")

    print("Saving results to results.csv")
    df.to_csv("results_sentiment.csv", index=False)

    print("Plotting visuals...")
    plot_sentiment_distribution(df)
    plot_sentiment_over_time(df)
    make_wordcloud_from_texts(df['clean_text'].tolist())

if __name__ == "__main__":
    main()
