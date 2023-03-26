import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from helper.preprocessing import Preprocessing
from helper.metrics import Metrics
from logistic.logisticRegression import LogisticRegression

df = pd.read_csv("../data/cancer.csv")
df = df.drop(df.columns[0],axis=1)

preprocessor = Preprocessing()

df = preprocessor.fillna(df)

y = df.iloc[:,0]
X = df.drop(df.columns[[0]],axis = 1)

y = preprocessor.categorical_to_numerical_l(y)

type = {
    0:"batch",
    1:"stochastic",
    2:"mini_batch"
}

res = pd.DataFrame()

epochs = 10000

for i in range(3):
    accuracy = 0
    precision = 0
    recall = 0
    
    X_train, X_test = preprocessor.train_test_split(X,train_size=0.67,random_state=13)
    y_train, y_test = preprocessor.train_test_split(y,train_size=0.67,random_state=13)

    X_train = preprocessor.normalize(X_train)
    logistic = LogisticRegression(0.0001,epochs,0.7,type[i],'not-normalized')
    if i == 0:
        logistic.fit_batch(X_train,y_train)
    elif i == 1:
        logistic.fit_stochastic(X_train,y_train)
    else:
        logistic.fit_mini_batch(X_train,y_train)

    X_test = preprocessor.normalize(X_test)
    y_pred = logistic.predict(X_test)
    y_test = np.array(y_test)

    metrics = Metrics(y_pred, y_test)
    accuracy=metrics.accuracy()
    precision=metrics.precision()
    recall=metrics.recall()
    print(accuracy,precision,recall,type[i])
