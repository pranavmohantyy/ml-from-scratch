import numpy as np
import pandas as pd
from datasets import DatasetLoader
from naive_bayes import GaussianNaiveBayes
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from random_forest import RandomForest
from decision_tree import DecisionTree
from k_nearest_neighbors import KNearestNeighbors
from svm_simple import LinearSVM
from metrics import Metrics
from cross_validation import CrossValidator

datasets = ['iris', 'wine', 'breast_cancer']
models = {
    'Naive Bayes': GaussianNaiveBayes(),
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForest(),
    'Decision Tree': DecisionTree(),
    'KNN': KNearestNeighbors(),
    'SVM': LinearSVM()
}

results = []

for dataset in datasets:
    loader = DatasetLoader(f'data/{dataset}.csv')
    data = loader.load_data()
    X, y = loader.split_data(target_column='target')
    cv = CrossValidator(n_splits=5)

    for model_name, model in models.items():
        scores = cv.k_fold(X.values, y.values, model)
        results.append((dataset, model_name, np.mean(scores)))

results_df = pd.DataFrame(results, columns=['Dataset', 'Model', 'Accuracy'])
print(results_df)
