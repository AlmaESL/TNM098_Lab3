import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pandas as pd

# make a corpus from the processed content of the dataset
def create_corpus(input_doc):
    # print("ID: ", input_doc['ID'])
    return input_doc['processed_content'].tolist()


# build a document array 
def build_tfidf_matrix(corpus):
    
   vectorizer = TfidfVectorizer(stop_words='english')
   tfidf_matrix = vectorizer.fit_transform(corpus)
   
   tfidf_df = pd.DataFrame(
       tfidf_matrix.toarray(), 
       columns=vectorizer.get_feature_names_out()
    )
   
   max_scores = tfidf_df.max(axis=0)
   top_40 = max_scores.sort_values(ascending=False).head(40)
   
   return top_40.items()

# vectorizer = TfidfVectorizer()



# function to normalize the tfidf matrix scores to 0-1 range
def normalize_tfidf_matrix(tfidf_matrix):
    # Normalize the TF-IDF matrix to a range of 0-1
    min_val = tfidf_matrix.min()
    max_val = tfidf_matrix.max()
    
    normalized_matrix = (tfidf_matrix - min_val) / (max_val - min_val)
    
    return normalized_matrix