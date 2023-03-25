import pandas as pd
import numpy as np


class Perceptron:
    def fit(self, X, y, epochs):
        # Initialization
        X["bias"] = 1
        X = X.to_numpy()
        X.reshape((X.shape[0],X.shape[1],1))
        y = y.to_numpy()
        self.weight = np.zeros_like(X[0])
        count = 0

        while count < epochs:
            misclassifications = 0
            for i in range(X.shape[0]):
                X_i = X[i]
                y_i = y[i]
                if y_i*(np.matmul(self.weight.T,X_i).item()) <= 0:
                    self.weight = self.weight+y_i*X_i
                    misclassifications += 1
            count+=1
            if misclassifications == 0:
                break

    def predict(self, X):
        X["bias"] = 1
        X = X.to_numpy()
        X.reshape((X.shape[0],X.shape[1],1))
        prediction = []
        for i in range(X.shape[0]):
            X_i = X[i]
            if np.matmul(self.weight.T, X_i).item() > 0:
                prediction.append(1)
            else:
                prediction.append(-1)
        return np.array(prediction)
