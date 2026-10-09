import numpy as np
from sklearn.model_selection import StratifiedKFold

class CrossValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def k_fold(self, X, y, model):
        fold_indices = np.array_split(np.arange(X.shape[0]), self.n_splits)
        scores = []
        for fold in fold_indices:
            X_train = np.delete(X, fold, axis=0)
            y_train = np.delete(y, fold, axis=0)
            X_val = X[fold]
            y_val = y[fold]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(self._calculate_score(y_val, y_pred))
        return np.mean(scores), np.std(scores)

    def stratified_k_fold(self, X, y, model):
        skf = StratifiedKFold(n_splits=self.n_splits)
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(self._calculate_score(y_val, y_pred))
        return np.mean(scores), np.std(scores)

    def _calculate_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred