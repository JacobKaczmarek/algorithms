import numpy as np
from sklearn.preprocessing import OneHotEncoder

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


class MulticlassLogisticRegression:
    def __init__(self, iters=1000, learning_rate=0.001, regularization=False):
        self.weights = None
        self.bias = None
        self.iters = iters
        self.learning_rate = learning_rate
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.regularization = regularization


    def fit(self, X, y):
        y_onehot = self.onehot_encoder.fit_transform(y.reshape(-1, 1))
        n_samples, n_features = X.shape
        C = y_onehot.shape[1]

        self.weights = np.zeros((n_features, C))

        for i in range(self.iters):
            Z = - X @ self.weights
            p = softmax(Z)
            dw = 1 / n_samples * X.T @ (y_onehot - p) + 2 * 1/(i+1) * self.weights
            print(dw)

            self.weights -= self.learning_rate * dw

            if (i % 100 == 0):
                print(f'Loss at {i} iteration: {self.loss(X, y_onehot)}')


    def loss(self, X, y):
        Z = - np.dot(X, self.weights)
        n_samples = X.shape[0]

        return (1 / n_samples) * (np.trace(X @ self.weights @ y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))


    def predict(self, X):
        Z = - X @ self.weights
        P = softmax(Z)

        return np.argmax(P, axis=1)
