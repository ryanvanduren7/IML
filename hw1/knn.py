import numpy as np
from download_mnist import load
import time
from collections import Counter

x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

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
        distances = [euclidean_dist(x, x_tr) for x_tr in self.X_train]
        # get k nearest samples, labels
        # argsort returns a list of indices
        k_ind = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_ind]
        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def euclidean_dist(x1, x2):
    return np.linalg.norm(x1 - x2)

def kNNClassify(newInput, dataSet, labels, k):
    clf = KNN(dataSet, labels, k)
    return clf.predict(newInput)

start_time = time.time()
outputlabels=kNNClassify(x_test[0:20],x_train,y_train,10)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
