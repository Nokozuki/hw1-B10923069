#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')


# In[3]:


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion="entropy")

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[4]:


# 將真正的答案和預測的答案整合到一個DataFrame
result_df = pd.DataFrame({
    '正確的類別': y_test,
    '預測的類別': y_pred
})

# 將DataFrame輸出到Excel檔案

with pd.ExcelWriter("ID3_model測試資料結果.xlsx", engine='openpyxl') as writer:
    result_df.to_excel(writer, index=False)

