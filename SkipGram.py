import torch
from torch import nn

class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        
        # complete this SkipGram model
        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim = 1)
        
    
    def forward(self, x):
        
        # define the forward behavior
        out = self.embed(x)
        out = self.output(out)
        return self.log_softmax(out)