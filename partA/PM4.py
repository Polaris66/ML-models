import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from classes.preprocessing import Preprocessing
from classes.perceptron import Perceptron
from classes.metrics import Metrics

df = pd.read_csv("../data/cancer.csv")
df.dropna(axis = 0,inplace=True)


preprocessor = Preprocessing()

y = df.iloc[:,1]
y = preprocessor.categorical_to_numerical(y)
X = df.drop(df.columns[[0,1]],axis = 1)


X_train, X_test = preprocessor.train_test_split(X,train_size=0.67,random_state=7)
y_train, y_test = preprocessor.train_test_split(y,train_size=0.67,random_state=7)

random_list = np.random.permutation(len(X_train.columns))
X_train = X_train.iloc[:,random_list]
X_test = X_test.iloc[:,random_list]

perceptron = Perceptron()
X_train = preprocessor.normalize(X_train)
perceptron.fit(X_train,y_train)

y_pred = perceptron.predict(X_test)
y_test = preprocessor.categorical_to_numerical(y_test)
metrics = Metrics()
misclassifications = metrics.misclassifications(y_pred,y_test)
print(misclassifications)