import numpy as np

class VotingClassifier:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.array([np.bincount(pred).argmax() for pred in predictions.T])

class StackingClassifier:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        self.predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            self.predictions[:, i] = model.predict(X)
        self.meta_model.fit(self.predictions, y)

    def predict(self, X):
        base_predictions = np.array([model.predict(X) for model in self.base_models]).T
        return self.meta_model.predict(base_predictions)