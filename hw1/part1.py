import numpy as np
from collections import Counter

class KNN:
    def __init__(self, X, y, k=3):
        self.k = k
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = np.array([self._predict(x) for x in X])
        return predicted_labels

    def _predict(self, x):
        # compute distances 
        distances = [euclidean_dist(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        # argsort returns a list of indices
        k_ind = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_ind]
        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def euclidean_dist(x1, x2):
    return np.linalg.norm(x1 - x2)

X_test = np.array([[1, 0, 1]])
X_train = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [1, 2, 1],
    [1, 2, 0],
    [1, 2, 2],
    [2, 2, 2],
    [1, 2, -1],
    [2, 2, 3],
    [-1, -1, -1],
    [0, -1, -2],
    [0, -1, -1],
    [-1, -2, 1]
    ])

y_test = np.array([0])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

for k in range(1, 4):
    clf = KNN(X_train, y_train, k)
    predictions = clf.predict(X_test)
    print("Predicted class ", predictions[0], " for k = ", k)
