import numpy as np

class NeuralNetwork:
    def __init__(self, W, X):
        self.W = W
        self.X = X

        # intermediate values
        self.product = self.matMult()
        self.sigmaVals = self.sigma()
        self.ans = self.loss()
        print("ans : ", self.ans)
        self.gr = None
        self.gp = None
        self.gX = None
        self.gW = None

    # calculates W * X
    def matMult(self):
        return np.matmul(self.W, self.X)

    # Element wise sigmoid calculation
    def sigma(self):
        j = 0
        temp = [0, 0, 0]
        for i in self.product:
            temp[j] = (1 / (1 + np.exp(-i)))
            j=j+1
        return temp

    def loss(self):
        return np.linalg.norm(self.sigmaVals)

    def backPropagate(self):
        self.gr = np.multiply(self.sigmaVals, 2)
        print("df/dr : ", self.gr)
        self.gp = []

        for element in self.gr:
            self.gp.append(np.exp(-element) / (1 + np.exp(-element)) ** 2)
        print("df/dp = ", self.gp)

        self.gX = np.matmul(2 * np.transpose(W), self.gp)
        print("df/dX : ", self.gX)

        self.gW = np.matmul(2 * np.asmatrix(self.gp).T, np.asmatrix(self.X))
        print("df/dW : ", self.gW)

W = np.array([[1, 0, 0],
              [2, 1, 0],
              [3, 5, 1]])

X = np.array([1, 0, 4])


s = NeuralNetwork(W, X)
s.backPropagate()

