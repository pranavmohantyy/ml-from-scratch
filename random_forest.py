import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=100, max_features='sqrt'): 
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_trees):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            else:
                max_features = n_features
            features_indices = np.random.choice(n_features, max_features, replace=False)
            tree = DecisionTree()
            tree.fit(X_bootstrap[:, features_indices], y_bootstrap)
            self.trees.append((tree, features_indices))

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X[:, features]) for tree, features in self.trees])
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)
        return majority_votes
