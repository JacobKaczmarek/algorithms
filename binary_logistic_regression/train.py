from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

dataset = datasets.load_breast_cancer()

X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(iters=100000, learning_rate=0.00001)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

