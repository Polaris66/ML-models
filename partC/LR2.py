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

epochs = 100

for i in range(3):
    for lr in [0.01,0.001,0.0001]:
        for thres in [0.3,0.4,0.5,0.6,0.7]:
            accuracy = 0
            precision = 0
            recall = 0
            for j in range(10):
                X_train, X_test = preprocessor.train_test_split(X,train_size=0.67,random_state=j+10)
                y_train, y_test = preprocessor.train_test_split(y,train_size=0.67,random_state=j+10)

                X_train = preprocessor.normalize(X_train)
                logistic = LogisticRegression(lr,epochs,thres,type[i],'normalized')
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
                accuracy+=metrics.accuracy()
                precision+=metrics.precision()
                recall+=metrics.recall()
            accuracy/=10
            precision/=10
            recall/=10
            row = {"model" : [type[i]],"learning rate" : [lr],"probability threshold" : [thres],"accuracy" : [accuracy],"precision" : [precision],"recall" : [recall],"epochs" : [epochs]}
            row = pd.DataFrame(row)
            print(row)
            res = pd.concat([res,row])
        logistic.plot()
        
        res.to_csv("LR2.csv",header=False)