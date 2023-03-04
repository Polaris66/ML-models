import pandas as pd
import numpy as np

class Preprocessing:
    def fillna(self,df):
        for col in df.columns:
            if col=="diagnosis":
                df[col].fillna(value = df[col].mode(),inplace=True)
            elif col!="id":
                df[col].fillna(value = df[col].mean(),inplace=True)
        return df

    def normalize(self,df):
        df.iloc[:,3:] =  (df.iloc[:,3:] - df.iloc[:,3:].mean())/df.iloc[:,3:].std()
        return df
    
    def train_test_split(self,df,train_size=0.67,test_size = 0.33,random_state=0):
        train = df.sample(frac= 0.67,random_state = random_state)
        test = df.drop(train.index)
        return train, test