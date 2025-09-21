from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOP
from typing import List
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def build_lda_model(token_lists: List[List[str]], num_topics: int = 6, passes: int = 10):
    # Create dictionary
    dictionary = corpora.Dictionary(token_lists)
    # Filter extremes
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in token_lists]
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)
    return lda, corpus, dictionary

def get_topics_as_list(lda_model, dictionary, topn=10):
    topics = []
    for tid in range(lda_model.num_topics):
        terms = lda_model.show_topic(tid, topn=topn)
        topics.append([t for t, w in terms])
    return topics
