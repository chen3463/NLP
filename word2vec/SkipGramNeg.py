
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

from SkipGramNeg import SkipGramNeg
from SkipGramNeg import NegativeSamplingLoss


def load_data(dir):

	# read in the extracted text file      
    with open(dir) as f:
    	text = f.read()

    # get list of words
    words = preprocess(text, True, False, False, True, 0)

    vocab_to_int, int_to_vocab = create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    train_words, freqs = subsampling(int_words, 1e-5, 100)

    return int_to_vocab, vocab_to_int, train_words, freqs

def train_SkipGramNeg(model, train_words, batches_size, int_to_vocab, embedding_dim, 
                   device, criterion, optimizer, epochs, print_every, data_dir):
	
	steps = 0
	# train for some number of epochs

	for e in range(epochs):
	    
	    # get our input, target batches
	    for input_words, target_words in get_batches(train_words, batches_size):
	        steps += 1
	        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
	        inputs, targets = inputs.to(device), targets.to(device)
	        
	        # input, outpt, and noise vectors
	        input_vectors = model.forward_input(inputs)
	        output_vectors = model.forward_output(targets)
	        noise_vectors = model.forward_noise(inputs.shape[0], 5)

	        # negative sampling loss
	        loss = criterion(input_vectors, output_vectors, noise_vectors)

	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()

	        # loss stats
	        if steps % print_every == 0:
	            print("Epoch: {}/{}".format(e+1, epochs))
	            print("Loss: ", loss.item()) # avg batch loss at this point in training
	            valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
	            _, closest_idxs = valid_similarities.topk(6)

	            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
	            for ii, valid_idx in enumerate(valid_examples):
	                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
	                print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
	            print("...\n")

	# getting embeddings from the embedding layer of our model, by name
	embeddings = model.in_embed.weight.to('cpu').data.numpy()
	# save embedding
	with open(os.path.join(data_dir, 'embeddings_neg.pkl'), "wb") as f:
         pickle.dump(embeddings, f)




if __name__ == '__main__':

	data_dir = '/content/drive/MyDrive/Colab_Notebooks/data/'
	int_to_vocab, vocab_to_int, train_words, freqs = load_data(os.path.join(data_dir, 'text8'))
	# check if GPU is available
	if torch.cuda.is_available():
		device = 'cuda' 
		print('using GPU')
	else: 
	 	device = 'cpu'
	 	print('using CPU')

	embedding_dim=300 

	# Get our noise distribution
	# Using word frequencies calculated earlier in the notebook
	word_freqs = np.array(sorted(freqs.values(), reverse=True))
	unigram_dist = word_freqs/word_freqs.sum()
	noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

	# instantiating the model
	model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device)

	# using the loss that we defined
	criterion = NegativeSamplingLoss() 
	optimizer = optim.Adam(model.parameters(), lr=0.003)

	print_every = 1500
	epochs = 5
	batches_size = 512
	

	train_SkipGramNeg(model, train_words, batches_size, int_to_vocab, embedding_dim, device, criterion, optimizer, epochs, print_every, data_dir)






