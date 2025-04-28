from __future__ import division
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import plotly.express as px
from tfidf import create_corpus

def plot_topic_temporal_distribution(raw_df, n_topics=9):
    # --- 1) Preprocess & vectorize ---
    corpus = create_corpus(raw_df)  
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(corpus)

    # --- 2) Fit LDA ---
    lda_model = LDA(n_components=n_topics, random_state=42)
    lda_model.fit(X)

    # --- 3) Assign main topic to each doc ---
    doc_topic_dists = lda_model.transform(X)
    main_topics = doc_topic_dists.argmax(axis=1) + 1   
    df = raw_df.copy()
    df['MainTopic'] = main_topics.astype(str)
    df['DateOnly']   = pd.to_datetime(df['Date']).dt.date

    # --- 4) Aggregate counts per day & topic ---
    agg = (
        df
        .groupby(['DateOnly','MainTopic'])
        .size()
        .reset_index(name='Count')
    )
    # ensure sorted by date
    agg = agg.sort_values('DateOnly')

    # --- 5) Plot with grouped bars, custom ordering & ticks ---
    # define the exact order of topics in the legend
    topic_order = [str(i) for i in range(n_topics+1)]

    fig = px.bar(
        agg,
        x='DateOnly',
        y='Count',
        color='MainTopic',
        barmode='group',                       # side-by-side bars
        category_orders={'MainTopic': topic_order},
        title='Number of Articles per Topic Over Time'
    )

    # thicker bars: reduce the gaps
    fig.update_layout(
        bargap=0.025,        # gap between groups of bars
        bargroupgap=0.0125,  # gap between bars within one group
        legend_title='Topic',
        legend_traceorder='normal'  # respect the category_orders sequence
    )

    # force every day to appear, full date format, rotated
    fig.update_xaxes(
        tickformat='%b %d',
        dtick='D1',        # one day interval
        tickangle=45
    )

    fig.update_yaxes(
        title_text='Article Count',
        tickmode='linear',  # Force integer tick mode
        dtick=1  # Set tick step to 1 to force integer ticks
    )

    fig.show()
