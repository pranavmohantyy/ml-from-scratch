import numpy as np
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression

class FeatureSelection:
    def __init__(self):
        self.selected_features = None

    def correlation_based(self, X, y, threshold=0.5):
        cor = X.corrwith(y)
        self.selected_features = cor[abs(cor) > threshold].index.tolist()
        return self.selected_features

    def mutual_information(self, X, y):
        mi = mutual_info_regression(X, y)
        self.selected_features = X.columns[mi > 0].tolist()
        return self.selected_features

    def recursive_feature_elimination(self, X, y, n_features):
        model = LinearRegression()
        selector = RFE(model, n_features)
        selector = selector.fit(X, y)
        self.selected_features = X.columns[selector.support_].tolist()
        return self.selected_features