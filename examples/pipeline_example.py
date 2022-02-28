"""
============================
Sklearn Pipeline Example
============================

An example of using the sklearn train_test_split pipeline with :class:`trees.StrongTree.StrongTreeClassifier`
"""
from pickle import TRUE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from trees.StrongTree import StrongTreeClassifier
from trees.utils.StrongTreeUtils import print_tree  

data = pd.read_csv("./data/balance-scale_enc.csv")
y = data.pop("target")

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.33, random_state=42
)

stcl = StrongTreeClassifier(
    depth = 1, 
    time_limit = 10,
    _lambda = 0,
    benders_oct= False, 
    num_threads=None, 
    obj_mode = 'acc'
)

stcl.fit(X_train, y_train)
print_tree(stcl.grb_model, stcl.b_value, stcl.w_value, stcl.p_value)
test_pred = stcl.predict(X_test)
print(np.sum(test_pred==y_test)/y_test.shape[0])
