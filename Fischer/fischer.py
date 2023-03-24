import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class fischer:

    def fit(self, X, y):

        # Initializing the attributes of model
        self.m1 = np.zeros(X.shape[1])
        self.m2 = np.zeros(X.shape[1])
        self.s1 = 0
        self.s2 = 0
        self.n1 = 0
        self.n2 = 0
        self.weight = np.zeros(X.shape[1])

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

        # # Diagnosing error
        # print("AVG OF CLASS 1:")
        # print(self.m1)
        # print("AVG OF CLASS 2:")
        # print(self.m2)

        # WORKING TILL HERE!!!!!!!!!!!!!!

        # Calculating the sb matrix
        self.sb = np.dot((self.m2.T - self.m1.T), (self.m2-self.m1).T)

        # Calculating the sw matrix
        self.sw = np.ones_like(self.sb)
        for i in range(X.shape[0]):
            if(y.iloc[i] == 1):
                self.sw += np.dot((X.iloc[i]-self.m1), (X.iloc[i]-self.m1).T)
            else:
                self.sw += np.dot((X.iloc[i]-self.m2), (X.iloc[i]-self.m2).T)

        # Find the direction of w
        self.weight = np.dot(np.linalg.inv(self.sw), self.m2-self.m1)

        # Finding the class variances
        for i in range(X.shape[0]):
            if(y.iloc[i] == 1):
                self.s1 += np.dot(self.weight,
                                  X.iloc[i])-np.dot(self.weight, self.m1)**2
            else:
                self.s2 += np.dot(self.weight,
                                  X.iloc[i])-np.dot(self.weight, self.m2)**2
        self.s1 = np.sqrt(self.s1)
        self.s2 = np.sqrt(self.s2)

        self.point = self.generative(np.dot(self.weight, self.m1),
                                     np.dot(self.weight, self.m2),
                                     self.s1,
                                     self.s2,
                                     self.n1,
                                     self.n2
                                     )

        self.plot(X, y)

    def plot(self, X, y):
        list1 = []
        list2 = []
        for i in range(X.shape[0]):
            if y.iloc[i] == 1:
                list1.append(
                    np.dot(self.weight, X.iloc[i])
                )
            else:
                list2.append(
                    np.dot(self.weight, X.iloc[i])
                )

        my_points_1 = np.prod(list1)
        my_points_2 = np.prod(list2)
        plt.bar(['Projected Points'], [my_points_1], color='red')
        plt.bar(['Projected Points'], [my_points_2], color='blue')
        plt.show()

        # Find Decision Boundary using Generative Approach
        # Too difficult :")

    def generative(self, m1, m2, s1, s2, n1, n2):
        a = s2**2-s1**2
        b = 2*m1*(s2**2)-2*m2*(s1**2)
        c = (s2**2)*(m1**2)-(s1**2)*(m2**2) - 2 * \
            (s1**2)*(s2**2)*np.log((n1*s2)/(n2*s1))
        return np.roots([a, b, c])[0]


df = pd.read_csv("feature_engineering_2.csv")
# df = pd.read_csv("data.csv")
df = df.dropna()

tr = df.iloc[:375, :]
test = df.iloc[375:, :]
y = tr.iloc[:, 1]
y_test = test.iloc[:, 1]

y_test.replace('M', 1, inplace=True)
y_test.replace('B', 0, inplace=True)

tr = tr.drop(tr.columns[[0, 1]], axis=1)
test = test.drop(test.columns[[0, 1]], axis=1)
y.replace('M', 1, inplace=True)
y.replace('B', 0, inplace=True)


model = fischer()
model.fit(tr, y)
