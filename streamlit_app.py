import streamlit as st
import pandas as pd
from preprocess import preprocess_dataframe
from sentiment_analysis import attach_sentiment
from visualizations import make_wordcloud_from_texts, plot_sentiment_distribution
import matplotlib.pyplot as plt
st.set_page_config(layout="wide", page_title="Social Media Analysis")

st.title("Social Media Analysis — Demo")
st.markdown("Upload CSV of posts (columns: content, date) or collect via script and upload results_sentiment.csv")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=['date'], low_memory=False)
    st.write("Raw data sample:")
    st.dataframe(df.head(5))

    if 'clean_text' not in df.columns:
        df = preprocess_dataframe(df)
        df = attach_sentiment(df)

    st.subheader("Sentiment distribution")
    fig, ax = plt.subplots()
    df['sentiment_label'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("Wordcloud")
    make_wordcloud_from_texts(df['clean_text'].tolist())

    st.subheader("Sample negative posts")
    st.write(df[df['sentiment_label']=='NEGATIVE'][['date','username','content']].head(10))
else:
    st.info("Upload a CSV to get started.")
