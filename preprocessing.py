import numpy as np

class StandardScaler:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class MinMaxScaler:
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.range_ = np.max(X, axis=0) - self.min_

    def transform(self, X):
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LabelEncoder:
    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, y):
        return np.array([self.class_to_index[c] for c in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

class OneHotEncoder:
    def fit(self, y):
        self.classes_ = np.unique(y)

    def transform(self, y):
        one_hot = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(y):
            one_hot[i, np.where(self.classes_ == c)[0][0]] = 1
        return one_hot

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)