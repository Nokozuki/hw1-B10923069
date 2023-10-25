#!/usr/bin/env python
# coding: utf-8

# In[10]:


from weka.classifiers import Classifier,Evaluation
from weka.core.converters import Loader
import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.core.dataset import Instances
from weka.core.serialization import read
import weka.core.serialization as serialization
import numpy as np
import pandas as pd
# 開啟jvm
jvm.start()
# 載入Arff資料，包含訓練與測試集
loader = Loader(classname="weka.core.converters.ArffLoader")
train_data = loader.load_file("train.arff")
train_data.class_index = train_data.num_attributes - 1
test_data = loader.load_file("test.arff")
test_data.class_index = test_data.num_attributes - 1
# model = Classifier(jobject=serialization.read('c4.5.model'))

# 使用weka中j48演算法
clr = Classifier(classname="weka.classifiers.trees.J48")
clr.build_classifier(train_data)
# 預測資料並評估
pred = Evaluation(train_data)
pred.test_model(clr, test_data)
print(pred.summary())



# In[ ]:



# 獲取預測結果
predictions = pred.predictions

# 提取正確的類別和分類器預測的類別
actual_classes = [prediction.actual for prediction in predictions]
predicted_classes = [prediction.predicted for prediction in predictions]

# 建立pandas DataFrame
result_df = pd.DataFrame({
    '正確的類別': actual_classes,
    '預測的類別': predicted_classes
})

# 輸出到Excel檔案
with pd.ExcelWriter("C4.5測試資料結果.xlsx", engine='openpyxl') as writer:
    result_df.to_excel(writer, index=False)

jvm.stop()

