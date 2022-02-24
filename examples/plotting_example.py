import pandas as pd
from sklearn.model_selection import train_test_split
from trees.StrongTree import StrongTreeClassifier
from trees.utils.TreePlotter import TreePlotter


data = pd.read_csv("data/balance-scale_enc.csv")
y = data.pop("target")

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.33, random_state=42
)

stcl = StrongTreeClassifier(4, 20, 0.6)

stcl.fit(X_train, y_train)


tree_plot = TreePlotter(
    stcl.tree,
    stcl.labels,
    stcl.X_col_labels,
    stcl.b_value,
    stcl.beta_value,
    stcl.p_value,
)

# TO-DO: need better way to choose node positions in drawNode()
tree_plot.plot()
