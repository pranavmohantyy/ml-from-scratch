import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.variance_ratio = None

    def fit(self, X):
        # compute the covariance matrix
        cov_matrix = np.cov(X, rowvar=False)
        # eigendecomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov_matrix)
        # sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]
        # explained variance ratio
        self.variance_ratio = self.eigenvalues / np.sum(self.eigenvalues)

    def transform(self, X):
        if self.n_components is not None:
            X = X @ self.eigenvectors[:, :self.n_components]
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)