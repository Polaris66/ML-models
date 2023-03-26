import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from helper.preprocessing import Preprocessing
from perceptron.perceptron import Perceptron
from helper.metrics import Metrics

df = pd.read_csv("../data/cancer.csv")
df.dropna(axis = 0,inplace=True)


preprocessor = Preprocessing()

y = df.iloc[:,1]
y = preprocessor.categorical_to_numerical(y)
X = df.drop(df.columns[[0,1]],axis = 1)


accuracy = 0
precision = 0
recall = 0
epochs = 10000

for i in range(10,20):
    X_train, X_test = preprocessor.train_test_split(
        X, train_size=0.67, random_state=i)
    y_train, y_test = preprocessor.train_test_split(
        y, train_size=0.67, random_state=i)

    perceptron = Perceptron()
    perceptron.fit(X_train, y_train,epochs)

    y_pred = perceptron.predict(X_test)
    y_test = preprocessor.categorical_to_numerical(y_test)
    y_test = y_test.to_numpy()

    metrics = Metrics(y_pred, y_test)
    accuracy+=metrics.accuracy()
    precision+=metrics.precision()
    recall+=metrics.recall()

accuracy/=10
precision/=10
recall/=10

series = pd.Series([accuracy,precision,recall, epochs],index=["accuracy","precision","recall", "epochs"])
print(series)
series.to_csv("PM2.csv", header=False)