import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, iters=1000, learning_rate=0.001):
        self.weights = None
        self.bias = None
        self.iters = iters
        self.learning_rate = learning_rate

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iters):
            linear_p = np.dot(X, self.weights) + self.bias
            p = sigmoid(linear_p)

            dw = (1 / n_samples) * np.dot(X.T, (p - y))
            db = (1/n_samples) * np.sum(p - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        pred = sigmoid(linear_pred)

        return [0 if y < 0.5 else 1 for y in pred]