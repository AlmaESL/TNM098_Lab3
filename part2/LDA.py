from __future__ import division
import sklearn 
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pyLDAvis
import pyLDAvis.graphlab
# import pyLDAvis.sklearn

from tfidf import create_corpus


def LDA_function(corpus): 
    corpus = create_corpus(corpus)
    
    vectorizer = CountVectorizer(stop_words='english', max_df =0.95, min_df=2)
    X = vectorizer.fit_transform(corpus)
    
    lda_model = LDA(n_components=5, random_state=42)
    lda_model.fit(X)
    
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.sklearn.prepare(lda_model, X, vectorizer, mds='tsne')
    panel
    
    
    
    
    
