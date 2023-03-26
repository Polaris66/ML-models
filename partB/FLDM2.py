import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from helper.preprocessing import Preprocessing
from helper.metrics import Metrics
from fischer.fischer import Fischer

df = pd.read_csv("../data/cancer.csv")
df = df.drop(df.columns[0],axis=1)

preprocessor = Preprocessing()

df = preprocessor.fillna(df)

y = df.iloc[:,0]
X = df.drop(df.columns[[0]],axis = 1)

y = preprocessor.categorical_to_numerical(y)

X = preprocessor.random_permutation(X)
accuracy = 0
precision = 0
recall = 0

for i in range(10):
    X_train, X_test = preprocessor.train_test_split(X,train_size=0.67,random_state=i)
    y_train, y_test = preprocessor.train_test_split(y,train_size=0.67,random_state=i)

    X_train = preprocessor.normalize(X_train)

    fischer = Fischer()
    fischer.fit(X_train,y_train,1)


    X_test = preprocessor.normalize(X_test)
    y_pred = fischer.predict(X_test)
    y_test = np.array(y_test)
    metrics = Metrics(y_pred, y_test)
    accuracy+=metrics.accuracy()
    precision+=metrics.precision()
    recall+=metrics.recall()
    fischer.plot(X_train, y_train,2,i)

accuracy/=10
precision/=10
recall/=10

series = pd.Series([accuracy,precision,recall, 'N/A'],index=["accuracy","precision","recall", "epochs"])
print(series)
series.to_csv("FLDM2.csv", header=False)