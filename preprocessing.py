from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
dataset = pd.read_csv('C:/Users/a6415/OneDrive/桌面/data mining/adult1.csv')
# 將缺值?改為None
dataset = dataset.replace(' ?',None)
dataset

dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
# 設定column的名子
dataset.set_axis(['age','workclass','fnlwgt','education','education-num',
                   'marital-status','occupation','relationship','race',
                   'sex','capital-gain','capital-loss','hours-per-week','native-country',
                   'income'],axis="columns",inplace=True)

# 將native-country缺值補為眾數aka United-States
dataset['native-country'].mode()
dataset = dataset.fillna({'native-country':' United-States'})
# 刪除其他缺失值
dataset = dataset.dropna()
dataset.info()

# 將不是數值的資料編碼為數字
# 注意!Sklearn的模型並不接收除numeric以外的資料，C4.5 C5.0則沒有此限
label_encoder = LabelEncoder()
dataset['income'] = label_encoder.fit_transform(dataset['income'])
dataset['workclass']=label_encoder.fit_transform(dataset['workclass'])
dataset['education']=label_encoder.fit_transform(dataset['education'])
dataset['marital-status']=label_encoder.fit_transform(dataset['marital-status'])
dataset['occupation']=label_encoder.fit_transform(dataset['occupation'])
dataset['relationship']=label_encoder.fit_transform(dataset['relationship'])
dataset['race']=label_encoder.fit_transform(dataset['race'])
dataset['sex']=label_encoder.fit_transform(dataset['sex'])
dataset['native-country']=label_encoder.fit_transform(dataset['native-country'])

dataset
# 輸出
dataset.to_csv('adult1.csv',index=False)
