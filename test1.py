import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from data_process import data_feature_engineering
from evaluate_model import evaluate_model_1

data_train = pd.read_csv("D:\Competition\Happiness\data\happiness_train_abbr_new.csv")
train_np = data_feature_engineering(data_train)#处理数据，将数据转换成0，1格式和归一化数据
print("train data process end")
y = train_np[:,0]#将第一列的happiness数据取出来当作标签
#y.to_csv("D:\Competition\Happiness\data\y.csv",index =False)
x = train_np[:, 1:]#将后面的数据取出来当作输入
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)#将数据分割成训练集和测试集
lr = LogisticRegression()
lr.fit(x_train,y_train)
r = lr.score(x_train, y_train)
print("R值(准确率):", r)
y_predict = lr.predict(x_test)  # 预测
p_r = evaluate_model_1(y_test,y_predict)
print(p_r)
flag = 0
for i in range(0,len(y_predict)):#统计测试集的预测结果
    if y_predict[i]==y_test[i]:
        flag = flag+1
print(flag/len(y_predict))
s = DataFrame({'y_predict':y_predict,'y_test':y_test})#转换成dataframe的格式进行存储
#s.to_csv("D:\Competition\Happiness\data\\t.csv",index =False)

"""开始对测试集进行预测"""
data_test = pd.read_csv("D:\Competition\Happiness\data\happiness_test_abbr_new.csv")
d_h = [0]*data_test.shape[0]#为了统一数据处理过程，为测试集加上全0的happiness这列数据
d_id = list(range(data_test['id'][0],data_test['id'][0]+data_test.shape[0]))
a = DataFrame({'id':d_id,'happiness':d_h})
data_test = pd.merge(a,data_test,on='id')
test_np = data_feature_engineering(data_test)#进行数据处理
X_test = test_np[:,1:]
y_predict_test = lr.predict(X_test)  # 预测
print(y_predict_test)
s = DataFrame({'happiness':y_predict_test})
print(s.shape[0])
s.to_csv("D:\Competition\Happiness\data\esult.csv",index =False)



