import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.means = None
        self.variances = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = np.array([np.mean(y == c) for c in self.classes])
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.variances = np.array([X[y == c].var(axis=0) for c in self.classes])

    def predict(self, X):
        log_probs = self._compute_log_likelihood(X)
        return self.classes[np.argmax(log_probs, axis=1)]

    def _compute_log_likelihood(self, X):
        log_likelihood = []
        for idx, c in enumerate(self.classes):
            mean = self.means[idx]
            var = self.variances[idx]
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_likelihood.append(log_prob + np.log(self.class_priors[idx]))
        return np.array(log_likelihood).T
