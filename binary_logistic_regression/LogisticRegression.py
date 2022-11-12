import numpy as np
from tqdm import tqdm

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
        losses = []

        for i in tqdm(range(self.iters)):
            linear_p = np.dot(X, self.weights) + self.bias
            p = sigmoid(linear_p)

            dw = (1 / n_samples) * np.dot(X.T, (p - y))
            db = (1/n_samples) * np.sum(p - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            losses.append(self.loss(X, y))

        return losses


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        pred = sigmoid(linear_pred)

        return [0 if y < 0.5 else 1 for y in pred]


    def loss(self, X, y):
        linear_p = np.dot(X, self.weights) + self.bias
        p = sigmoid(linear_p) + 0.0001

        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))