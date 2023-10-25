#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')


# In[45]:


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


# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# 找到ccp_alpha值最佳值
best_index = np.argmax(test_scores)
best_clf = clfs[best_index]

print("Best ccp_alpha:", ccp_alphas[best_index])
print("Train accuracy:", train_scores[best_index])
print("Test accuracy:", test_scores[best_index])


# In[47]:


best_alpha_index = np.where(ccp_alphas == ccp_alphas[best_index])[0][0]
best_alpha = ccp_alphas[best_alpha_index]
print(best_alpha, best_alpha_index)


# In[48]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# ccp_alpha值：最小值、中間值和最佳值
alpha_values = [ccp_alphas[0], ccp_alphas[len(ccp_alphas)//2], ccp_alphas[best_alpha_index]]

clfs = []
for ccp_alpha in alpha_values:
    clf = DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# 繪製樹
for i, clf in enumerate(clfs):
    plt.figure(figsize=(20, 10))

    class_names_str = [str(cls) for cls in clf.classes_]
    
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=class_names_str, rounded=True)
    plt.title(f"Decision Tree with ccp_alpha: {alpha_values[i]}")
    plt.show()

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for ccp_alpha {alpha_values[i]}:", accuracy)


# In[11]:



# 為每個 ccp_alpha 值訓練決策樹
trees = [DecisionTreeClassifier(criterion="entropy", ccp_alpha=alpha).fit(X_train, y_train) for alpha in alpha_values]

# 獲取每棵樹的節點數和深度
node_counts = [tree.tree_.node_count for tree in trees]
depths = [tree.tree_.max_depth for tree in trees]

# 輸出結果
for i, alpha in enumerate(alpha_values):
    print(f"For ccp_alpha = {alpha}:")
    print(f"Number of nodes: {node_counts[i]}")
    print(f"Depth of tree: {depths[i]}\n")


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label="Train", drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label="Test", drawstyle="steps-post")
plt.xlabel("Alpha values")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Alpha for training and testing sets")
plt.legend()
plt.grid(True)
plt.show()

