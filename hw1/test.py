import numpy as np


def euclidean_dist(x1, x2):
    return np.linalg.norm(x1 - x2)


x_test = np.array([1, 0, 1])

A = np.array([[0, 1, 0],
              [0, 1, 1],
              [1, 2, 1],
              [1, 2, 0]])

B = np.array([[1, 2, 2],
              [2, 2, 2],
              [1, 2, -1],
              [2, 2, 3]])

C = np.array([[-1, -1, -1],
              [0, -1, -2],
              [0, -1, -1],
              [-1, -2, 1]])

X_train = np.array([
    [[0, 1, 0],
     [0, 1, 1],
     [1, 2, 1],
     [1, 2, 0]],

    [[1, 2, 2],
     [2, 2, 2],
     [1, 2, -1],
     [2, 2, 3]],

    [[-1, -1, -1],
     [0, -1, -2],
     [0, -1, -1],
     [-1, -2, 1]]
])


list1 = np.array([1, 3, 4,2,6,4,20,69])
print(np.argsort(list1))
