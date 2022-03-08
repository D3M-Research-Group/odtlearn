"""
============================
StrongTree Fit Example
============================

An example of fitting a StrongTree decision tree using :class:`odtlearn.StrongTree.StrongTreeClassifier`
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from odtlearn.StrongTree import StrongTreeClassifier

data = pd.read_csv("./data/balance-scale_enc.csv")
y = data.pop("target")

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.33, random_state=42
)

stcl = StrongTreeClassifier(
    depth=1,
    time_limit=60,
    _lambda=0,
    benders_oct=False,
    num_threads=None,
    obj_mode="acc",
)

stcl.fit(X_train, y_train, verbose=True)
stcl.print_tree()
test_pred = stcl.predict(X_test)
print(
    "The out-of-sample acc is {}".format(np.sum(test_pred == y_test) / y_test.shape[0])
)
