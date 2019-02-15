import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def data_feature_engineering(data):
    data.loc[data['family_income'].isnull(), 'family_income'] = 0#避免含有0的情况
    data.drop(['work_status', 'work_yr', 'work_type', 'work_manage'], axis=1, inplace=True)#这几个却是的太多，直接剔除
    #print("123")
    data['age'] = 2019-data['birth']#计算年龄
    data_line = data.shape[0]#将行数存储下来
    data_1 = pd.read_csv("D:\Competition\Happiness\data\happiness_train_abbr_new.csv")#读取test和train 的数据，和输入数据一起做一个dummy，这样可以保证dummy的数量相同
    data_2 = pd.read_csv("D:\Competition\Happiness\data\happiness_test_abbr_new.csv")
    data_1['age'] = 2019-data_1['birth']
    data_2['age'] = 2019-data_2['birth']
    data_1.loc[data_1['family_income'].isnull(), 'family_income'] = 0
    data_2.loc[data_2['family_income'].isnull(), 'family_income'] = 0
    data_1.drop(['work_status', 'work_yr', 'work_type', 'work_manage'], axis=1, inplace=True)
    data_2.drop(['work_status', 'work_yr', 'work_type', 'work_manage'], axis=1, inplace=True)
    #print("123")
    d_h = [0] * data_2.shape[0]
    d_id = list(range(data_2['id'][0], data_2['id'][0] + data_2.shape[0]))
    a = DataFrame({'id': d_id, 'happiness': d_h})
    data_2 = pd.merge(a, data_2, on='id')
    #print("123")

    data_3 = pd.concat([data_1,data_2],ignore_index = True)

    data5 = pd.concat([data,data_3],ignore_index = True )

   # print("456")





    survey_type_dummies = pd.get_dummies(data5['survey_type'],prefix = 'survey')
    gender_dummies = pd.get_dummies(data5['gender'], prefix='gender')
    province_dummies = pd.get_dummies(data['province'],prefix='province')

    nationality_dummies = pd.get_dummies(data5['nationality'], prefix='nationality')
    religion_dummies = pd.get_dummies(data5['religion'], prefix='religion')
    edu_dummies = pd.get_dummies(data5['edu'], prefix='edu')
    religion_freq_dummies = pd.get_dummies(data5['religion_freq'], prefix='religion_freq')
    political_dummies = pd.get_dummies(data5['political'], prefix='political')
    health_dummies = pd.get_dummies(data5['health'], prefix='health')
    health_problem_dummies = pd.get_dummies(data5['health_problem'], prefix='health_problem')
    depression_dummies = pd.get_dummies(data5['depression'], prefix='depression')
    socialize_dummies = pd.get_dummies(data5['socialize'], prefix='socialize')
    relax_dummies = pd.get_dummies(data5['relax'], prefix='relax')
    learn_dummies = pd.get_dummies(data5['learn'], prefix='learn')
    equity_dummies = pd.get_dummies(data5['equity'], prefix='equity')
    family_status_dummies = pd.get_dummies(data5['family_status'], prefix='family_status')
    class_dummies = pd.get_dummies(data5['class'], prefix='class')
    car_dummies = pd.get_dummies(data5['car'], prefix='car')
    marital_dummies = pd.get_dummies(data5['marital'], prefix='marital')
    status_peer_dummies = pd.get_dummies(data5['status_peer'], prefix='status_peer')
    status_3_before_dummies = pd.get_dummies(data5['status_3_before'], prefix='status_3_before')
    view_dummies = pd.get_dummies(data5['view'], prefix='view')
    inc_ability = pd.get_dummies(data5['inc_ability'], prefix='inc_ability')
   # print("456")




    df = pd.concat([data, survey_type_dummies,gender_dummies,nationality_dummies,religion_dummies, edu_dummies,religion_freq_dummies,
                    political_dummies,health_dummies, health_problem_dummies,depression_dummies,socialize_dummies,relax_dummies,
                    learn_dummies,equity_dummies,family_status_dummies,class_dummies, car_dummies,marital_dummies,
                    status_peer_dummies, status_3_before_dummies, view_dummies, inc_ability,province_dummies], axis=1)



    df.drop(['survey_type','gender','nationality','religion','edu','religion_freq','political','health','health_problem',
             'depression','socialize','relax','learn','equity','family_status','class','car','marital','status_peer',
             'status_3_before','view','inc_ability','province'], axis=1, inplace=True)#扔掉dumm的数据
    #print('789')
    df = df.iloc[0:data_line,:]#将要处理的数据取出来，test和train不要


    scaler = preprocessing.StandardScaler()

    age_scale_param = scaler.fit(data_1['age'].values.reshape(-1, 1))

    df['Age_scaled'] = scaler.fit_transform(df['age'].values.reshape(-1, 1), age_scale_param)
   # print('233')


    age_scale_param = scaler.fit(data_1['income'].values.reshape(-1, 1))
    df['income_scaled'] = scaler.fit_transform(df['income'].values.reshape(-1, 1), age_scale_param)
    age_scale_param = scaler.fit(data_1['floor_area'].values.reshape(-1, 1))
    df['floor_area_scaled'] = scaler.fit_transform(df['floor_area'].values.reshape(-1, 1), age_scale_param)
    age_scale_param = scaler.fit(data_1['height_cm'].values.reshape(-1, 1))
    df['height_cm_scaled'] = scaler.fit_transform(df['height_cm'].values.reshape(-1, 1), age_scale_param)
    age_scale_param = scaler.fit(data_1['weight_jin'].values.reshape(-1, 1))
    df['weight_jin_scaled'] = scaler.fit_transform(df['weight_jin'].values.reshape(-1, 1), age_scale_param)
    age_scale_param = scaler.fit(data_1['family_income'].values.reshape(-1, 1))
    df['family_income_scaled'] = scaler.fit_transform(df['family_income'].values.reshape(-1, 1), age_scale_param)
    age_scale_param = scaler.fit(data_1['family_m'].values.reshape(-1, 1))
    df['family_m_scaled'] = scaler.fit_transform(df['family_m'].values.reshape(-1, 1), age_scale_param)
   # df.drop(['age','income','floor_area','height_cm','weight_jin','family_income','family_m'], axis=1, inplace=True)

    train_df = df.filter(regex='happiness|survey_type_.*|gender_.*|nationality_.*|religion_.*|religion_freq_.*|political_.*'
                               '|health_.*|health_problem_.*|depression_.*|socialize_.*|relax_.*|equity_.*|family_status_.*|family_m_.*'
                               '|class_.*|car_.*|marital_.*|status_peer_.*|status_3_before_.*|view_.*|inc_ability_.*|province_.*|Age_scaled|income_scaled|floor_area_scaled|height_cm_scaled|weight_jin_scaled|family_m_scaled')
    #print(train_df.columns.values.tolist())
    #train_df.info()
    print("i am here")
    #train_df
    #train_df.to_csv("D:\Competition\Happiness\data\y.csv", index=True)
    train_np = train_df.as_matrix()
   # print("i am here two")
    return train_np

    pass
