import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit


class Logistic_Regression_batch:
    def __init__(self, lr, iter, thres):
        self.lr = lr
        self.iter = iter
        self.thres = thres
        self.weight = None
        self.history = None
        return

    def sigmoid(self, X):
        a = expit(np.dot(X, self.weight))
        return a

    def classify(self, prob):
        if prob >= self.thres:
            return 1
        else:
            return 0

    def gradient(self, X, Y):
        grad_E = np.zeros(X.shape[1])
        cnt = 0
        for n in range(X.shape[0]):
            t_n = Y[n]
            X_n = X.iloc[n, :]
            y_n = self.sigmoid(X_n)
            # Gradient of Error function
            grad_E += (y_n-t_n)*X_n
            cnt += 1
        return grad_E/cnt

    def cost(self, X, Y):
        cst = 0.
        for n in range(X.shape[0]):
            t_n = Y[n]
            X_n = X.iloc[n, :]
            y_n = self.sigmoid(X_n)
            if t_n == 1:
                t_n -= 0.000001
            if y_n == 1:
                y_n -= 0.000001
            # Gradient of Error function
            cst += t_n*np.log(y_n)+(1-t_n)*np.log(1-y_n)
            print("Cost: "+str(cst))
        cst *= -1.
        return cst

    def fit(self, X, Y):
        # Add a bias column of all 1s
        X["bias"] = 1
        # Initialize the weights to 0
        self.weight = np.ones(X.shape[1])
        self.history = []
        # Run the gradient descent algorithm
        for i in range(self.iter):
            # Update the weight based on the gradient with the current weight vector
            self.weight -= self.lr*self.gradient(X, Y)
            # Count misclassifications
            misclassifications = 0
            # Print the misclassifications
            if i % 10 == 0 or i < 10:
                for j in range(X.shape[0]):
                    X_j = X.iloc[j, :]
                    Y_j = Y.iloc[j]
                    if self.classify(self.sigmoid(X_j)) != Y_j:
                        misclassifications += 1
                self.history.append((i, self.cost(X, Y)))
                print(self.weight)
                print("misclassifications:")
                print(misclassifications)
                print("epochs:")
                print(i)
        return

    def predict(self, X):
        X["bias"] = 1
        prediction = []
        for i in range(X.shape[0]):
            X_i = X.iloc[i, :]
            prediction.append(self.classify(self.sigmoid(X_i)))
        return prediction


class Logistic_Regression_stochastic:
    def __init__(self, lr, iter):
        self.lr = lr
        self.iter = iter
        self.weight = None
        self.history = None
        return

    def sigmoid(self, X):
        a = expit(np.dot(X, self.weight))
        return a

    def classify(self, prob):
        if prob >= 0.5:
            return 1
        else:
            return 0

    def fit(self, X, Y):
        # Add a bias column of all 1s
        X["bias"] = 1
        # Initialize the weights to 0
        self.weight = np.ones(X.shape[1])
        self.history = []
        # Run the gradient descent algorithm
        for i in range(self.iter):
            # Update the weight based on the gradient with the current weight vector
            for n in range(X.shape[0]):
                grad_E = np.zeros(X.shape[1])
                t_n = Y[n]
                X_n = X.iloc[n, :]
                y_n = self.sigmoid(X_n)
                # Gradient of Error function
                grad_E += (y_n-t_n)*X_n
                self.weight -= self.lr*grad_E
            # Count misclassifications
            misclassifications = 0
            # Print the misclassifications
            if i % 10 == 0 or i < 10:
                for j in range(X.shape[0]):
                    X_j = X.iloc[j, :]
                    Y_j = Y.iloc[j]
                    if self.classify(self.sigmoid(X_j)) != Y_j:
                        misclassifications += 1
                self.history.append((i, self.cost(X, Y)))
                print(self.weight)
                print("misclassifications:")
                print(misclassifications)
                print("epochs:")
                print(i)
        return

    def predict(self, X):
        X["bias"] = 1
        prediction = []
        for i in range(X.shape[0]):
            X_i = X.iloc[i, :]
            prediction.append(self.classify(self.sigmoid(X_i)))
        return prediction

    def cost(self, X, Y):
        cst = 0.
        for n in range(X.shape[0]):
            t_n = Y[n]
            X_n = X.iloc[n, :]
            y_n = self.sigmoid(X_n)
            if t_n == 1:
                t_n -= 0.000001
            if y_n == 1:
                y_n -= 0.000001
            # Gradient of Error function
            cst += t_n*np.log(y_n)+(1-t_n)*np.log(1-y_n)
            print("Cost: "+str(cst))
        cst *= -1.
        return cst


class Logistic_Regression_mini_batch:
    def __init__(self, lr, iter):
        self.lr = lr
        self.iter = iter
        self.weight = None
        self.history = None
        return

    def sigmoid(self, X):
        a = expit(np.dot(X, self.weight))
        return a

    def classify(self, prob):
        if prob >= 0.5:
            return 1
        else:
            return 0

    def fit(self, X, Y):
        # Add a bias column of all 1s
        X["bias"] = 1
        # Initialize the weights to 0
        self.weight = np.ones(X.shape[1])
        self.history = []
        # Run the gradient descent algorithm
        for i in range(self.iter):
            # Update the weight based on the gradient with the current weight vector
            grad_E = np.zeros(X.shape[1])
            ind = 0
            for n in range(X.shape[0]):
                ind += 1
                t_n = Y[n]
                X_n = X.iloc[n, :]
                y_n = self.sigmoid(X_n)
                # Gradient of Error function
                grad_E += (y_n-t_n)*X_n
                if n % 20 == 0:  # Batch strength is 20
                    self.weight -= self.lr*grad_E/ind
                    grad_E = np.zeros(X.shape[1])
                    ind = 0
                elif n == X.shape[0]-1:
                    self.weight -= self.lr*grad_E/ind
                    ind = 0
            # Count misclassifications
            misclassifications = 0
            # Print the misclassifications
            if i % 10 == 0 or i < 10:
                for j in range(X.shape[0]):
                    X_j = X.iloc[j, :]
                    Y_j = Y.iloc[j]
                    if self.classify(self.sigmoid(X_j)) != Y_j:
                        misclassifications += 1
                self.history.append((i, self.cost(X, Y)))
                print(self.weight)
                print("misclassifications:")
                print(misclassifications)
                print("epochs:")
                print(i)
        return

    def predict(self, X):
        X["bias"] = 1
        prediction = []
        for i in range(X.shape[0]):
            X_i = X.iloc[i, :]
            prediction.append(self.classify(self.sigmoid(X_i)))
        return prediction

    def cost(self, X, Y):
        cst = 1
        for n in range(X.shape[0]):
            t_n = Y[n]
            X_n = X.iloc[n, :]
            y_n = self.sigmoid(X_n)
            if t_n == 1:
                t_n -= 0.01
            if y_n == 1:
                y_n -= 0.01
            # Gradient of Error function
            cst *= np.log10((t_n*np.log(y_n)+(1-t_n)*np.log(1-y_n)))
            print("Cost: "+str(cst))
        return cst