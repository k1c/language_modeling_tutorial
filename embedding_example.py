import torch
import torch.nn as nn

# Define our hyperparameters
N = 10 # The number of words/tokens in our dictionary/vocabulary
ninp = 3 # The dimensionality of our word embeddings. So each word-vector will be a 3-tuple [x_0, x_1, x_2].
E = nn.Embedding(num_embeddings=N,embedding_dim=ninp) # embedding_dim: size of the vector space that you want each word to live in

# Always make sure you understand the shape of every vector, matrix, tensor/etc you are dealing with:
# print(E.weight.size()) # It’s look up table, what size should it be?
print('E:', E.weight) # Look at E.weight
X = torch.LongTensor([1,2,4,5]) # (What is X?)
E(X)
print('X:', X, 'E(X):', E(X))
