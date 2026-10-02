import numpy as np

class LogisticRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, learning_rate=0.01, n_iterations=1000, l2_penalty=0.0):
        m, n = X.shape
        self.coef_ = np.random.randn(n)
        for _ in range(n_iterations):
            linear_model = X @ self.coef_
            y_predicted = self.sigmoid(linear_model)
            gradient = (1 / m) * X.T @ (y_predicted - y) + (l2_penalty / m) * self.coef_
            self.coef_ -= learning_rate * gradient

    def predict(self, X):
        linear_model = X @ self.coef_
        y_predicted = self.sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binary_cross_entropy(self, y_true, y_pred):
        m = len(y_true)
        return - (1 / m) * np.sum(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))