import numpy as np
import random
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

from bs4 import BeautifulSoup

import torch

def preprocess(text, is_replace_punctuation, is_remove_stopwords, is_stem, is_remove, remove_threshold):

    if is_replace_punctuation:

        # Replace punctuation with tokens so we can use them in our model
        text = text.lower()
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

    text = BeautifulSoup(text, "html.parser").get_text() # Remove HTML tags

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case

    words = text.split()

    if is_remove_stopwords:
        nltk.download("stopwords", quiet=True)
        words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords

    if is_stem:
        stemmer = PorterStemmer()
        words = [PorterStemmer().stem(w) for w in words] # stem

    # Remove all words with  5 or fewer occurences
    if is_remove:
        word_counts = Counter(words)
        if remove_threshold is None:
           remove_threshold = 5 
        clean_words = [word for word in words if word_counts[word] > remove_threshold]
    else:
        clean_words = words

    return clean_words


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

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



