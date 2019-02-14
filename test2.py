import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data_t = pd.read_csv("D:\Competition\Happiness\data\happiness_train_abbr.csv")

data = data_t
for i in range (data.shape[1]):
    print(i)
    if data.columns[i]!='survey_time'and data.columns[i]!='gender':
        for j in range(data.shape[0]):
            if data.ix[j][i]<0:
                data.iloc[j,i] = 0
    if data.columns[i]=='gender'or data.columns[i]=='religion':
        for j in range(data.shape[1]):
            if data.ix[j][i]<0:
                data.iloc[j,i] = -1


data.to_csv("D:\Competition\Happiness\data\happiness_train_abbr_new.csv",index =False)