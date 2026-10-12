import numpy as np

class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        poly_features = [X]
        for d in range(2, self.degree + 1):
            poly_features.append(X ** d)
        return np.hstack(poly_features)

    def transform(self, X):
        return self.fit_transform(X)