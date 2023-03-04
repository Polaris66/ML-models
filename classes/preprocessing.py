{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Preprocessing:\n",
    "    def fillna(self,df):\n",
    "        for col in df.columns:\n",
    "            if col==\"diagnosis\":\n",
    "                df[col].fillna(value = df[col].mode(),inplace=True)\n",
    "            elif col!=\"id\":\n",
    "                df[col].fillna(value = df[col].mean(),inplace=True)\n",
    "        return df\n",
    "\n",
    "    def normalize(self,df):\n",
    "        df.iloc[:,3:] =  (df.iloc[:,3:] - df.iloc[:,3:].mean())/df.iloc[:,3:].std()\n",
    "        return df\n",
    "    \n",
    "    def train_test_split(df,train_size = 0.67, test_size=0.33,random_state=0):\n",
    "        train = df.sample(frac= test_size,random_state = random_state)\n",
    "        test = df.drop(train.index)\n",
    "        return train, test"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
