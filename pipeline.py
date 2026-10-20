import numpy as np

class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        for name, step in self.steps:
            X = step.fit_transform(X, y)
        return self

    def predict(self, X):
        for name, step in self.steps[:-1]:
            X = step.transform(X)
        return self.steps[-1][1].predict(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)