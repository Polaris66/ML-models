import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit


class LogisticRegression:
    def __init__(self, lr, epochs, thres, type, norm):
        self.lr = lr
        self.epochs = epochs
        self.thres = thres
        self.weight = None
        self.history = None
        self.type = type
        self.norm = norm
        return

    def fit_batch(self, X, y):
        # Add a bias column of all 1s
        X["bias"] = 1
        X = X.to_numpy()
        X.reshape((X.shape[0],X.shape[1],1))
        y = y.to_numpy()
        y.reshape((y.shape[0],1))
        
        # Initialize the weights to 0
        self.weight = np.ones_like(X[0])
        self.history = []
        # Run the gradient descent algorithm
        for i in range(self.epochs):
            # Update the weight based on the gradient with the current weight vector
            self.weight -= self.lr*self.gradient_batch(X, y)
            self.history.append((i, self.cost(X, y)))
        return


    def fit_stochastic(self, X, y):
        # Add a bias column of all 1s
        X["bias"] = 1
        X = X.to_numpy()
        X.reshape((X.shape[0],X.shape[1],1))
        y = y.to_numpy()
        y.reshape((y.shape[0],1))
        
        # Initialize the weights to 0
        self.weight = np.ones_like(X[0])
        self.history = []
        # Run the gradient descent algorithm
        for i in range(self.epochs):
            # Update the weight based on the gradient with the current weight vector
            for j in range(X.shape[0]):
                grad = np.zeros(X.shape[1])
                t_j = y[j]
                X_j = X[j]
                y_j = self.sigmoid(X_j)
                # Gradient of Error function
                grad += (y_j-t_j)*X_j
                self.weight -= self.lr*grad
            self.history.append((i, self.cost(X, y)))
        return
    
    def fit_mini_batch(self, X, y):
        # Add a bias column of all 1s
        X["bias"] = 1
        X = X.to_numpy()
        X.reshape((X.shape[0],X.shape[1],1))
        y = y.to_numpy()
        y.reshape((y.shape[0],1))
        
        # Initialize the weights to 0
        self.weight = np.ones_like(X[0])
        self.history = []
        grad = np.zeros(X.shape[1])
        # Run the gradient descent algorithm
        for i in range(self.epochs):
            # Update the weight based on the gradient with the current weight vector
            for j in range(X.shape[0]):
                if j % 10 == 0:
                    self.weight -= self.lr*grad
                    grad = np.zeros(X.shape[1])
                t_j = y[j]
                X_j = X[j]
                y_j = self.sigmoid(X_j)
                # Gradient of Error function
                grad += (y_j-t_j)*X_j
            self.history.append((i, self.cost(X, y)))
        return


    def predict(self, X):
        X["bias"] = 1
        prediction = []
        for i in range(X.shape[0]):
            X_i = X.iloc[i, :]
            prediction.append(self.classify(self.sigmoid(X_i)))
        return np.array(prediction)

    def sigmoid(self, X):
        a = expit(np.matmul(self.weight.T,X).item())
        return a

    def classify(self, prob):
        if prob >= self.thres:
            return 1
        else:
            return 0

    def gradient_batch(self, X, y):
        grad = np.zeros(X.shape[1])
        n = 0
        for i in range(X.shape[0]):
            t_i = y[i]
            X_i = X[i]
            y_i = self.sigmoid(X_i)
            # Gradient of Error function
            grad += (y_i-t_i)*X_i
            n += 1
        return grad/n

    def cost(self, X, y):
        c = 0.
        for i in range(X.shape[0]):
            t_i = y[i]
            X_i = X[i]
            y_i = self.sigmoid(X_i)
            if y_i < 10**(-15):
                y_i+=10**(-15)
            elif y_i > 1-10**(-15):
                y_i-=1-10**(-15)
            # Gradient of Error function
            c += t_i*np.log(y_i)+(1-t_i)*np.log(1-y_i)
        c *= -1.
        return c
    
    def plot(self):        
        points = self.history

        # extract x and y values from each tuple
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        plt.clf()
        # plot the points
        plt.plot(x, y, 'ro')

        # add axis labels and a title
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.title(self.type)

        # display the plot
        plt.savefig('./graphs/'+str(self.type)+'-'+str(self.lr)+'-'+self.norm+'.png')
