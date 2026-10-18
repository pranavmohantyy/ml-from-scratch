import numpy as np
from sklearn.utils import resample

class SMOTE:
    def __init__(self, sampling_strategy='auto', random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        # Implement SMOTE logic here
        return X_resampled, y_resampled

class RandomUnderSampler:
    def __init__(self, sampling_strategy='auto', random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        # Implement random undersampling logic here
        return X_resampled, y_resampled