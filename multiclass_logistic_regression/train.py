from sklearn import datasets
from sklearn.model_selection import train_test_split
from MulticlassLogisticRegression import MulticlassLogisticRegression
from sklearn.metrics import accuracy_score

dataset = datasets.load_digits()

X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MulticlassLogisticRegression(iters=5, learning_rate=0.001)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))