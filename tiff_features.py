"""
Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval 
and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. 
The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word 
in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a 
document's relevance given a user query.

One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions
are variants of this simple model.

Tf-idf can be successfully used for stop-words filtering in various subject fields including text summarization and classification.



Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a 
word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF),
computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.

TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible 
that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document 
length (aka. the total number of terms in the document) as a way of normalization: 

TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important.
However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need
to weigh down the frequent terms while scale up the rare ones, by computing the following: 

IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

Source from http://www.tfidf.com
"""
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    ####### YOUR CODE HERE #######
    tfidf_vectorizer = TfidfVectorizer(token_pattern = '(\S+)', ngram_range=(1,2), max_df = 0.9, min_df = 5)
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_
