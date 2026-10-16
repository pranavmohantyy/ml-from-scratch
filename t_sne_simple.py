import numpy as np
from sklearn.manifold import TSNE

class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.embedding = None

    def fit_transform(self, X):
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, learning_rate=self.learning_rate)
        self.embedding = tsne.fit_transform(X)
        return self.embedding

    def transform(self, X):
        return self.fit_transform(X)