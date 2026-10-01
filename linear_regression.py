import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, method='normal'):
        if method == 'normal':
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            self.coef_ = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        else:
            self.gradient_descent(X, y)

    def gradient_descent(self, X, y, learning_rate=0.01, n_iterations=1000):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m = X_b.shape[0]
        self.coef_ = np.random.randn(X_b.shape[1])
        for _ in range(n_iterations):
            gradients = 2/m * X_b.T @ (X_b @ self.coef_ - y)
            self.coef_ -= learning_rate * gradients

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.coef_

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)