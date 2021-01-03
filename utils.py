import numpy as np
import random
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords", quiet=True)
nltk.download('wordnet', quiet=True)

from bs4 import BeautifulSoup
import torch
import string        

def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text() # Remove HTML tags

def remove_urls(text):
    return re.compile(r'https?://\S+|www\.\S+').sub(r'',text)
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punctuation(text):
    no_punct = ""
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in text:
       if char not in punctuations:
           no_punct = no_punct + char

    return no_punct

def replace_puncutation(text):
    # Replace punctuation with tokens so we can use them in our model
    # text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
        # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')    

    return text 

def remove_stopwords(words):
    
    return [w for w in words if w not in stopwords.words("english")] # Remove stopwords

def lemmatize_words(words):
    
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def stemmer_words(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in words] # stem

def clean_text(text, punctuation, is_remove_stopwords, stem_or_lem, is_remove_less, 
    num_remove_words):
    
    text = remove_urls(text) # Removing urls
    text = remove_html(text) # Removing HTML tags
    text = remove_emoji(text)

    text = text.lower()

    if punctuation == 'replace':
        text = replace_puncutation(text)
        
    else:
        text = remove_punctuation(text)

    text = re.sub(r"[^a-zA-Z0-9]", " ", text) # Convert to lower case

    words = text.split()

    if is_remove_stopwords:
        words = remove_stopwords(words)     

    if stem_or_lem == 'stem':
        words = stemmer_words(words)

    elif stem_or_lem == 'lem':
        words = lemmatize_words(words)

    else:
        raise Exception("Error input of stem or lem")

    # Remove all words with  5 or fewer occurences
    if is_remove_less:
        word_counts = Counter(words)
        if num_remove_words is None:
            num_remove_words = 5 
        clean_words = [word for word in words if word_counts[word] > num_remove_words]
    else:
        clean_words = words

    return clean_words

def clean_text_list():
    pass


def create_corpus(text_list):
    corpus=[]
    for text in text_list:
        words=[word.lower() for word in word_tokenize(text) if word.isalpha()==1]
        corpus.append(words)
    return corpus

def create_lookup_tables(text, vocab_size = None):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words or list of list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    
    if isinstance(text[0], str):
      word_counts = Counter(text)
    elif isinstance(text[0], list):
      word_counts = {} # A dict storing the words that appear in the reviews along with how often they occur
      for sentence in text:
          for word in sentence:
              if word not in word_counts.keys():
                  word_counts[word] = 1
              else:
                  word_counts[word] += 1
    else:
      raise Exception("Words input is wrong")
    
    if vocab_size is None:
        vocab_size = len(word_counts) + 2
    # sorting the words from most to least frequent in text occurrence
    word_count_sorted = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {} # This is what we are building, a dictionary that translates words into integers
    int_to_vocab = {}
    for idx, word in enumerate(word_count_sorted[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        vocab_to_int[word] = idx + 2  
        int_to_vocab[idx + 2] = word   

    return vocab_to_int, int_to_vocab

def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """
    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities

def get_target(words, idx, window_size = 5, seed = 100):
    ''' Get a list of words in a window around an index. '''
    
    random.seed(seed)
    # implement this function
    R = random.randint(1, window_size + 1)
    if idx - R < 0:
        start = 0
    else:
        start = idx - R 
    
    end = idx + R
    
    return words[start : idx] + words[idx + 1 : end + 1]

def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words) // batch_size
    
    # only full batches
    words = words[: (n_batches * batch_size)]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx : (idx + batch_size)]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y

def subsampling(int_words, threshold, seed):
    '''
    discard words with probability given by 

    P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}

    where $t$ is a threshold parameter and $f(w_i)$ is the frequency of word $w_i$ in the total dataset.
    '''
    word_counts = Counter(int_words)
    total_count = len(int_words)
    freqs = {int_word: freq / total_count for (int_word, freq) in word_counts.items()}
    p_drops = {int_word: 1 - np.sqrt(threshold / freqs[int_word]) for int_word in word_counts}

    random.seed(seed)

    train_words = [int_word for int_word in int_words if random.random() < (1 - p_drops[int_word])]

    return train_words, freqs


