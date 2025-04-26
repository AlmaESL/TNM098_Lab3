import pandas as pd
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load model for lemmatization
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

# Prepare stopwords
stop_words = set(stopwords.words('english'))

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

def load_and_process(filepath):
    # Load Dataset
    df = pd.read_csv(filepath, delimiter=';', encoding='utf-8')
    # Apply preprocessing
    df['processed_content'] = df['Content'].apply(preprocess)
    return df
