import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        m, n = X.shape
        random_indices = np.random.choice(m, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids
        return labels, self.inertia(X, labels)

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def inertia(self, X, labels):
        return sum(np.linalg.norm(X[labels == i] - self.centroids[i])**2 for i in range(self.n_clusters))
