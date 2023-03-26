import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm

class Fischer:

    def generative(self, m1, m2, s1, s2, n1, n2):
        a = 1/(2*s1**2) - 1/(2*s2**2)
        b = m2/(s2**2) - m1/(s1**2)
        c = m1**2 / (2*s1**2) - m2**2 / (2*s2**2) - np.log((n1*s2)/(n2*s1))
        return np.roots([a, b, c])

    def fit(self, X, y, k):

        # Initializing the attributes of model
        self.m1 = np.zeros(X.shape[1])
        self.m2 = np.zeros(X.shape[1])
        self.s1 = 0
        self.s2 = 0
        self.n1 = 0
        self.n2 = 0
        self.k = k
        # Finding the class means
        for i in range(X.shape[0]):
            if(y.iloc[i] == 1):
                self.m1 += X.iloc[i]
                self.n1 += 1
            else:
                self.m2 += X.iloc[i]
                self.n2 += 1
        self.m1 /= self.n1
        self.m2 /= self.n2
        self.m1 = self.m1.to_numpy()
        self.m1 = self.m1.reshape((self.m1.shape[0], 1))
        self.m2 = self.m2.to_numpy()
        self.m2 = self.m2.reshape((self.m2.shape[0], 1))

        # Calculating the sb matrix
        self.sb = np.matmul((self.m2 - self.m1), (self.m2-self.m1).T)
        # Calculating the sw matrix
        self.sw = np.zeros_like(self.sb)
        X = X.to_numpy()

        X = X.reshape((X.shape[0], X.shape[1], 1))
        for i in range(X.shape[0]):
            if(y.iloc[i] == 1):
                self.sw += np.matmul((X[i]-self.m1),
                                     (X[i]-self.m1).T)
            else:
                self.sw += np.matmul((X[i]-self.m2),
                                     (X[i]-self.m2).T)
        # Find the direction of w
        self.weight = np.matmul(
            np.linalg.inv(self.sw), (self.m2-self.m1))
        # Finding the class variances
        for i in range(X.shape[0]):
            if(y.iloc[i] == 1):
                self.s1 += (np.matmul(self.weight.T,
                                      X[i])-np.matmul(self.weight.T, self.m1))**2
            else:
                self.s2 += (np.matmul(self.weight.T,
                                      X[i])-np.matmul(self.weight.T, self.m2))**2
        self.s1 /= self.n1
        self.s2 /= self.n2
        self.s1 = np.sqrt(self.s1.item())
        self.s2 = np.sqrt(self.s2.item())
        self.m1p = np.matmul(self.weight.T, self.m1).item()
        self.m2p = np.matmul(self.weight.T, self.m2).item()
        self.point = self.generative(self.m1p,
                                     self.m2p,
                                     self.s1.item(),
                                     self.s2.item(),
                                     self.n1,
                                     self.n2
                                     )

    def predict(self, X):
        y = []
        X = X.to_numpy()
        X = X.reshape((X.shape[0], X.shape[1], 1))

        for i in range(X.shape[0]):
            x = np.matmul(self.weight.T, X[i])
            if(x.item() > self.point[self.k]):
                y.append(-1)
            else:
                y.append(1)
        return np.array(y)
    
    def plot(self, X, y, n, j):
        list1 = []
        list2 = []
        plt.clf()
        X = X.to_numpy()
        X = X.reshape((X.shape[0], X.shape[1], 1))
        for i in range(X.shape[0]):
            if y.iloc[i] == 1:
                list1.append(
                    np.matmul(self.weight.T, X[i]).item()
                )
            else:
                list2.append(
                    np.matmul(self.weight.T, X[i]).item()
                )
        my_points_1 = np.array(list1)
        my_points_2 = np.array(list2)
        y1 = np.zeros_like(my_points_1)+1
        plt.plot(my_points_1, y1, c='red')
        y2 = np.zeros_like(my_points_2)+2
        plt.plot(my_points_2, y2, c='blue')

        # (left, right) = (-0.05, 0.04)
        (left, right) = (-0.2, 0.1)
        # x1 = np.linspace(-0.05, 0.04, 10000)
        x1 = np.linspace(left, right, 10000)
        plt.plot(x1, -1*norm.pdf(
            x1, self.m1p, self.s1.item()), c='red')
        x1 = np.linspace(left, right, 10000)
        plt.plot(x1, -1*norm.pdf(
            x1, self.m2p, self.s2.item()), c='blue')
        # display the plot
        plt.savefig('./graphs/'+str(n)+'-'+str(j)+'.png')

        # Find Decision Boundary using Generative Approach
        # Too difficult :")

