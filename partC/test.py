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

df.dropna(axis = 0,inplace=True)

y = df.iloc[:,0]
X = df.drop(df.columns[[0]],axis = 1)

y = preprocessor.categorical_to_numerical_l(y)

X_train, X_test = preprocessor.train_test_split(X,train_size=0.67,random_state=1)
y_train, y_test = preprocessor.train_test_split(y,train_size=0.67,random_state=1)

logistic = LogisticRegression(lr,100,thres,type[i], 'not')
logistic.fit_batch(X_train,y_train)

y_pred = logistic.predict(X_test)
y_test = np.array(y_test)

logistic.plot()