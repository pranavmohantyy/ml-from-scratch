import numpy as np

class DecisionTree:
    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        num_samples, num_features = X.shape
        if num_samples == 0:
            return None
        if len(np.unique(y)) == 1:
            return y[0]

        best_feature = self._best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()

        left_indices = X[:, best_feature] < np.median(X[:, best_feature])
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices])
        right_tree = self._build_tree(X[right_indices], y[right_indices])

        return (best_feature, left_tree, right_tree)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        for feature in range(X.shape[1]):
            gain = self._calculate_gain(X[:, feature], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature

    def _calculate_gain(self, feature_values, y):
        # Calculate gain based on Gini or entropy
        # This is a placeholder for simplicity
        return np.random.rand()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, left_tree, right_tree = tree
        if x[feature] < np.median(x[feature]):
            return self._traverse_tree(x, left_tree)
        else:
            return self._traverse_tree(x, right_tree)
