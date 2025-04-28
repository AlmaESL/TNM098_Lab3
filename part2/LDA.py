from __future__ import division
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis
import pyLDAvis.lda_model as lda_vis
import webbrowser
from tfidf import create_corpus

def LDA_function(raw_corpus):
    # 1) build your list of documents
    corpus = create_corpus(raw_corpus)

    # 2) vectorize as plain counts
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(corpus)

    # 3) fit sklearn’s LDA
    lda_model = LDA(n_components=9, random_state=42)
    lda_model.fit(X)

    # 4) prepare the pyLDAvis panel
    panel = lda_vis.prepare(lda_model, X, vectorizer, mds='tsne')

    # 5) save and open
    output_path = 'lda_vis.html'
    pyLDAvis.save_html(panel, output_path)
    print(f"LDA visualization saved to {output_path}")
    webbrowser.open_new_tab(output_path)
