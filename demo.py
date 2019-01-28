"""A demonstration"""

# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

from data import *
from model import *

PATH = 'data/wikitext-2/'  #a bunch of wikipedia articles that have been flattened

wikitext = Corpus(PATH)

words = list(wikitext.dictionary.word2idx.keys())

for word in words[:10]:
    print(word, wikitext.dictionary.word2idx[word])

print('There are {} total words/tokens in our dictionary.'.format(len(words)))

for word in words[:10]:
    val = wikitext.dictionary.word2idx[word]
    print(val, wikitext.dictionary.idx2word[val])

# Look at the actual training data:
print(type(wikitext.train)) # Tensor
print(wikitext.train.size()) # 2,088,628 So Wikitext-2 is where we took a bunch of Wikipedia articles and flattened them out into a MASSIVE paragraph of text. It's a paragraph that consists of 2.1 million words, punctuation marks, etc.

# As of now, this is just exploratory data analysis. Now we want to get framework specific, we eventually want to feed this into a model and train it via minibatch-SGD. As of now, our dataset is just a 2.1M length vector.

# Now we jump to data pre-processing. The main objective here is to convert this huge 2.1M length vector into mini-batches. That's precisely what the batchify function does:

def batchify(data, bsz): #should say matrixify: produces a matrix form
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz    #data.size(0) = total length of the dataset. nbatch is equal to M where M=corpus size / batchsize
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous() #view in pytorch is like resize
    return data

BATCHSIZE = 20 # So we will feed in 20 different sentences at a time.

train_data = batchify(wikitext.train, BATCHSIZE)

print(train_data.size()) # 104431x20

# We have 20 "mega-sentences" of length 104,431. That is totally ridiculous. We will process each mega-sentence chunks at a time.

BPTT = 35 #Backprop through time

# Now our dataset is a matrix of size MxB, so we have B "sentences" of length M, where M is huge. We define a function that allows us to break up these M length sentences into BPTT-length sentences:
# The size of the sliding window is BPTT

def get_batch(source, i):
    seq_len = min(BPTT, len(source) - 1 - i) # Done for division purposes, the last chunk left may not be precisely of size BPTT!
    data = source[i:i+seq_len] # See what this is doing? It is getting the i-th chunk of BPTT-rows from our data matrix.
    target = source[i+1:i+1+seq_len].view(-1) # It is getting the next words, these are the labels for language modeling, and it's reshaping.
    return data, target

# So our minibatches will consists of 20 (BATCHSIZE) sentences of length 35 (BPTT).
# Pause here to see if this makes sense.
# All in all our model will be processing 20*35 = 700 total words in each batch, and it must make 700 predictions of the next word, so we will need 700 labels, and we will have 700 LOSSES!

batch_0 = get_batch(train_data, 0)
batch_0_x = batch_0[0]
batch_1 = get_batch(train_data, 1)
batch_1_x = batch_1[0]

# Print the first two datapoints:
print('first sentence', train_data[0])
print('second sentence', train_data[1])

# Notice that the first datapoint of batch_0 is indeed datapoint 0, and the first data_point of batch_1 is datapoint 1! So it is a sliding window.
print('first sentence of batch_0', batch_0_x[0])
print('first sentence of batch_1', batch_1_x[0])

print('last sentence of batch_0', batch_0_x[-1])
print('second to last sentence of batch_1', batch_1_x[-2])
print('third to last sentence of batch_1', batch_1_x[-3])
