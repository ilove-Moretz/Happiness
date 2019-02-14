import pandas as pd
import numpy as np
from pandas import Series,DataFrame
def evaluate_model_1(data_1,data_2):
    data_1_list = data_1.tolist()
    data_2_list = data_2.tolist()

    result = DataFrame({'lable':data_1_list,'predict':data_2_list})
    print(result.shape[0])
    p_r=[]
    for i in range(1,6):
        s1 = result.loc[result['predict'] ==i]
        p = s1.shape[0]
        s1 = s1.loc[result['lable']==i]
        tp = s1.shape[0]
        s1 = result.loc[result['lable'] == i]
        t = s1.shape[0]
        a=[tp/p,tp/t]
        p_r.append(a)
    return p_r





