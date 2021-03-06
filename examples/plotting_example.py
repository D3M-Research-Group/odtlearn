"""
============================
Tree Plotting Example
============================

An example of plotting a fit decision tree using :class:`odtlearn.utils.TreePlotter.TreePlotter`
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from odtlearn.StrongTree import StrongTreeClassifier
from odtlearn.RobustTree import RobustTreeClassifier
import matplotlib.pyplot as plt


data = pd.read_csv("data/balance-scale_enc.csv")
y = data.pop("target")

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.33, random_state=42
)


stcl = StrongTreeClassifier(
    depth=4,
    time_limit=5,
    _lambda=0.51,
    benders_oct=True,
    num_threads=None,
    obj_mode="acc",
)

stcl.fit(X_train, y_train)

stcl.plot_tree()
plt.show()

###################################

X = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 1],
    ]
)
y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])

costs = np.array(
    [
        [1, 4],
        [1, 4],
        [1, 4],
        [1, 4],
        [1, 4],
        [4, 4],
        [4, 4],
        [4, 4],
        [4, 4],
        [4, 4],
        [4, 1],
        [4, 4],
        [4, 4],
    ]
)


robust_classifier = RobustTreeClassifier(
    depth=2,
    time_limit=10,
)

robust_classifier.fit(X, y, costs=costs, budget=5, verbose=False)
robust_classifier.print_tree()

robust_classifier.plot_tree()
plt.show()
