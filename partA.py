import pandas as pd
import numpy as np
from classes.preprocessing import Preprocessing
from classes.perceptron import Perceptron

df = pd.read_csv("data/cancer.csv")
df.dropna(axis = 0,inplace=True)
y = df.iloc[:,1]
y.replace('B',1,inplace=True)
y.replace('M',-1,inplace=True)

X = df.drop(df.columns[[0,1]],axis = 1)

preprocessor = Preprocessing()

X_train, X_test = preprocessor.train_test_split(X,train_size=0.67,random_state=5)
y_train, y_test = preprocessor.train_test_split(y,train_size=0.67,random_state=5)

perceptron = Perceptron()
X_train = preprocessor.normalize(X_train)
perceptron.fit(X_train,y_train)
print(perceptron.weight)