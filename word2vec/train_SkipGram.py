
import os
import random
import numpy as np
import pickle

from collections import Counter

import torch
from torch import nn
import torch.optim as optim

from utils import get_batches
from utils import cosine_similarity
from utils import preprocess
from utils import create_lookup_tables
from utils import subsampling

from SkipGram import SkipGram



def load_data(dir):

	# read in the extracted text file      
    with open(dir) as f:
    	text = f.read()

    # get list of words
    words = preprocess(text, True, False, False, True, 5)

    vocab_to_int, int_to_vocab = create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    train_words, freqs = subsampling(int_words, 1e-5, 100)

    return int_to_vocab, vocab_to_int, train_words, freqs

def train_SkipGram(model, train_words, batches_size, int_to_vocab, embedding_dim, 
                   device, criterion, optimizer, epochs, print_every, data_dir):
	
	steps = 0
	# train for some number of epochs
	for e in range(epochs):
	    
	    # get input and target batches
	    for inputs, targets in get_batches(train_words, batches_size):
	        steps += 1
	        inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
	        inputs, targets = inputs.to(device), targets.to(device)
	        
	        log_ps = model(inputs)
	        loss = criterion(log_ps, targets)
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	
	        if steps % print_every == 0:  
	            print("Epoch: {}/{}".format(e+1, epochs))
	            print("Loss: ", loss.item()) # avg batch loss at this point in training    
	            valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)                 
	            _, closest_idxs = valid_similarities.topk(6) # topk highest similarities

	            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
	            for ii, valid_idx in enumerate(valid_examples):
	                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
	                print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
	            print("...")

	# getting embeddings from the embedding layer of our model, by name
	embeddings = model.embed.weight.to('cpu').data.numpy()
	# save embedding
	with open(os.path.join(data_dir, 'embeddings.pkl'), "wb") as f:
         pickle.dump(embeddings, f)
            
if __name__ == '__main__':

	data_dir = '/content/drive/MyDrive/Colab_Notebooks/data/'
	int_to_vocab, vocab_to_int, train_words, _ = load_data(os.path.join(data_dir, 'text8'))
	# check if GPU is available
	if torch.cuda.is_available():
		device = 'cuda' 
		print('using GPU')
	else: 
	 	device = 'cpu'
	 	print('using CPU')

	embedding_dim=300 

	model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.003)
	batches_size = 512
	print_every = 500
	epochs = 5

	train_SkipGram(model, train_words, batches_size, int_to_vocab, embedding_dim, device, criterion, optimizer, epochs, print_every, data_dir)




