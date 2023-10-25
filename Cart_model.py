#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

data = pd.read_csv('adult.data')
data.head()
data.drop(data.columns[[0]], axis=1, inplace=True)

X = data.drop('income',axis=1)
y = data['income']

y
X
# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立決策樹
clf = DecisionTreeClassifier(criterion="gini")
# 放入訓練資料
clf.fit(X_train, y_train)
# 預測結果
y_pred = clf.predict(X_test)

# 正確率計算
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[4]:


get_ipython().system('pip install openpyxl')


# In[3]:


import pandas as pd
import openpyxl

result_df = pd.DataFrame({
    '正確的類別': y_test,
    '預測的類別': y_pred
})

with pd.ExcelWriter("Cart_model測試資料結果.xlsx", engine='openpyxl') as writer:
    result_df.to_excel(writer, index=False)

