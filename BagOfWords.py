import nltk
import ssl
from ast import literal_eval
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english')+[''])

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text)# delete symbols which are in BAD_SYMBOLS_RE from text
    # print(text.split(' '))
    text = ' '.join([w for w in text.split(' ') if not w in STOPWORDS]) # delete stopwords (including '') from text
    print(text)
    return text

DICT_SIZE = 5000
common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE]
WORDS_TO_INDEX = {word:i for i, word in enumerate(sorted(word for word,count in common_words))} 
INDEX_TO_WORDS = {i:word for i, word in enumerate(sorted(word for word,count in common_words))} 
ALL_WORDS = WORDS_TO_INDEX.keys()
####### YOUR CODE HERE #######

for i in INDEX_TO_WORDS:
    WORDS_TO_INDEX[list(ALL_WORDS)[i]] = i
    

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)        
    for word in text.split(" "):
        if word in words_to_index:
            result_vector[words_to_index[word]] = 1
    return result_vector
    
