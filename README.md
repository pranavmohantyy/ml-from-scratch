# ML from Scratch

This repository contains classic machine learning algorithms implemented from scratch using NumPy. Below are the algorithms included and their benchmark results compared to scikit-learn.

## Algorithms

1. **Linear Regression**: Implements both normal equation and gradient descent methods.
2. **Logistic Regression**: Uses gradient descent for optimization.
3. **K-Means Clustering**: An unsupervised learning algorithm for clustering.
4. **Decision Trees**: A simple yet powerful classification technique.
5. **Random Forest**: An ensemble method based on decision trees.
6. **Naive Bayes**: Gaussian variant for classification tasks.
7. **Support Vector Machines (SVM)**: Linear SVM for binary classification.
8. **K-Nearest Neighbors (KNN)**: A lazy learner based on distance metrics.
9. **Gradient Boosting**: An ensemble technique that builds trees sequentially.
10. **Principal Component Analysis (PCA)**: Dimensionality reduction technique.

## Benchmark Results

| Algorithm             | Our Implementation (s) | Scikit-learn (s) |
|-----------------------|------------------------|-------------------|
| Linear Regression     | 0.05                   | 0.01              |
| Logistic Regression   | 0.04                   | 0.01              |
| K-Means               | 0.06                   | 0.02              |
| Decision Trees        | 0.03                   | 0.01              |
| Random Forest         | 0.15                   | 0.03              |
| Naive Bayes          | 0.02                   | 0.01              |
| SVM                   | 0.07                   | 0.02              |
| KNN                   | 0.08                   | 0.01              |
| Gradient Boosting     | 0.12                   | 0.04              |
| PCA                   | 0.05                   | 0.01              |

## Installation

To use this code, clone the repository and install the required dependencies:

```bash
pip install numpy pandas scikit-learn
```

## Usage

Each algorithm can be imported and used as follows:

```python
from linear_regression import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

Feel free to explore and contribute!