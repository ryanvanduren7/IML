import numpy as np
import matplotlib as mpl
from collections import Counter
import matplotlib.pyplot as plt
mpl.use('Agg')

# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')

# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)

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

# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    clf = KNN(dataSet, labels, k)
    return clf.predict(newInput)


outputlabels = kNNClassify(mini_test,mini_train,mini_train_label,4)

print ('random test points are:\n', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]

fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)],
train_y[np.where(mini_train_label==0)], color='red')

plt.scatter(train_x[np.where(mini_train_label==1)],
train_y[np.where(mini_train_label==1)], color='blue')

plt.scatter(train_x[np.where(mini_train_label==2)],
train_y[np.where(mini_train_label==2)], color='yellow')

plt.scatter(train_x[np.where(mini_train_label==3)],
train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]

outputlabels = np.array(outputlabels)

plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)],
marker='^', color='red')

plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)],
marker='^', color='blue')

plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)],
marker='^', color='yellow')

plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)],
marker='^', color='black')
plt.savefig("miniknnpic.png")

