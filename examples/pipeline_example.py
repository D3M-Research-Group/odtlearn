import pandas as pd
from sklearn.model_selection import train_test_split
from trees.StrongTree import StrongTreeClassifier


data = pd.read_csv("/Users/patrick/project-template/data/balance-scale_enc.csv")
y = data.pop("target")

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.33, random_state=42
)

stcl = StrongTreeClassifier(1, 100, 0)

stcl.fit(X_train, y_train)

stcl.predict(X_test)
