import numpy as np

class NeuralNetwork:
    def __init__(self, data=None):
        if data is None:
            data = []
        print("Data : ", data)
        self.x1 = data[0]
        self.gx1 = None
        self.x2 = data[1]
        self.gx2 = None
        self.w1 = data[2]
        self.gw1 = None
        self.w2 = data[3]
        self.gw2 = None

        # intermediate values
        self.p1 = None
        self.gp1 = None
        self.p2= None
        self.gp2 = None
        self.a1 = None
        self.ga1 = None
        self.a2 = None
        self.ga2 = None
        self.r = None
        self.gr = None
        self.ans = None
        self.gans = 1

    # calculates p1 for flag, else calculates p2
    def mult(self):
        self.p1 = self.x1 * self.w1
        self.p2 = self.x2 * self.w2

    # adds 2 + a1 + a2
    def add2(self):
        self.r = 2 + self.a1 + self.a2

    def signSquared(self):
        self.a1 = (np.sin(self.p1)) ** 2

    def cosine(self):
        self.a2 = np.cos(self.p2)

    def inverse(self):
        self.ans = 1 / self.r

    def compute(self):
        # compute p1 and p2
        self.mult()
        # compute a1 and a2
        self.signSquared(), self.cosine()
        # compute r
        self.add2()
        # compute and return ans
        self.inverse()
        return self.ans

    def backPropagate(self):
        self.gr = self.gans * (-1 / (self.r **2))
        self.gp1 = self.gr * np.sin(2 * self.p1)
        self.gp2 = self.gr * np.sin(self.p2) * -1

        self.gx1 = self.gp1 * self.w1
        print("gx1 = ", self.gx1)
        self.gw1 = self.gp1 * self.x1
        print("gw1 = ", self.gw1)

        self.gx2 = self.gp2 * self.w2
        print("gx2 = ", self.gx2)
        self.gw2 = self.gp2 * self.x2
        print("gw2 : ", self.gw2)

        return [self.gx1, self.gx2, self.gw1, self.gw2]

data1 = [1, 2, 3, 4]
NN = NeuralNetwork(data1)
NN.compute()
print(NN.backPropagate())
