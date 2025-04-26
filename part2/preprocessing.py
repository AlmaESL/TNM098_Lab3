import pandas as pd
import spacy
import re
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data files 
nltk.download('punkt')
nltk.download('stopwords')

# Load Dataset
df = pd.read_csv('C:/Users/almal/Desktop/termin8/TNM098/lab 3/TNM098_Lab3/TNM098-MC3-2011.csv', delimiter=';', encoding='utf-8')

# Quick check
# print(df.head())

stop_words = set(stopwords.words('english'))
# Load model for lemmatization
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')  

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    doc = nlp(' '.join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

df['processed_content'] = df['Content'].apply(preprocess)

# Quick look
# print(df[['Content', 'processed_content']].head())


# Temporal Distribution (Unfiltered)
df['Date'] = pd.to_datetime(df['Date'])

fig = px.histogram(df, x='Date', nbins=50, title='News Report Frequency Over Time (All Data)')
fig.show()
