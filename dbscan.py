import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = -1 * np.ones(n_samples)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                continue
            self.labels_[i] = cluster_id
            self._expand_cluster(X, i, neighbors, cluster_id, visited)
            cluster_id += 1

    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                new_neighbors = self._region_query(X, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.unique(np.append(neighbors, new_neighbors))
            if self.labels_[neighbor] == -1:
                self.labels_[neighbor] = cluster_id