import pandas as pd
import numpy as np

class Perceptron:
    def fit(self,X,y):
        #Initialization
        X["bias"] = 1
        self.weight= np.zeros(X.shape[1])

        while True:
            misclassifications = 0
            for i in range(X.shape[0]):
                X_i = X.iloc[i,:]
                y_i = y.iloc[i,:]
                if y_i*(np.dot(self.weight,X_i))<=0:
                    self.weight = self.weight+y_i*X_i
                    misclassifications+=1
            if misclassifications==0:
                break

    def predict(self,X):
        prediction = np.empty()
        for i in range(X.shape[0]):
            X_i = X.iloc[i,:]
            if np.dot(X_i,self.weight)>0:
                prediction.append(1)
            else:
                prediction.append(-1)
        return prediction