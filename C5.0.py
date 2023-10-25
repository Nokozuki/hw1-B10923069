from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from rpy2.robjects import numpy2ri
import numpy as np
r = robjects.r
# utils = importr('utils')
# utils.install_packages("C50")
C50=importr('C50')
caret = importr('caret')
C50_model = robjects.r("C5.0Control(subset=T, winnow=F, noGlobalPruning=T, minCases=20)")
data = pd.read_csv('adult1.csv')
data['income'] = pd.Categorical(data['income'])
X = data.drop('income',axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)

pandas2ri.activate()
r_train =pandas2ri.py2rpy(train)
r_test = pandas2ri.py2rpy(test)
model = C50.C5_0(robjects.Formula("income ~ ."),data=train,rules=False,control=C50_model)

pred = robjects.r.predict(model, newdata=test)

char_vec = robjects.r['as.vector'](pred)
py_pred = list(char_vec)

acc = accuracy_score(y_test, py_pred)
print('Accuracy:',acc)

op = pd.concat([pd.DataFrame(y_test).reset_index(drop=True),pd.DataFrame(py_pred)],axis=1)


op.columns = ['正確的結果','預測的結果']
with pd.ExcelWriter("C5.0測試資料結果.xlsx", engine='openpyxl') as writer:
    op.to_excel(writer, index=False)

