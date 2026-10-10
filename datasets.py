import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def split_data(self, target_column, test_size=0.2, random_state=None):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
