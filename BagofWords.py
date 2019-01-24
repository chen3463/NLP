import re
def extract_words(document, stopwords, stem=None, lax=None):
    words = re.sub("[^\w]", " ", document).split()
    words_clean = [words.lower() for w in words if w not in stopwords]
    
    return words_clean
    
