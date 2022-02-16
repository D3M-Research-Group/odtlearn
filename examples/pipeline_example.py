import pandas as pd
from sklearn.model_selection import train_test_split
from trees.FairTree import FairTreeClassifier


# data = pd.read_csv("../data/balance-scale_enc.csv")
# y = data.pop("target")
#
# X_train, X_test, y_train, y_test = train_test_split(
#     data, y, test_size=0.33, random_state=42
# )
#
# stcl = StrongTreeClassifier(1, 100, 0)
#
# stcl.fit(X_train, y_train)
#
# stcl.predict(X_test)


data = pd.read_csv("../data/compas/compas_train_1.csv")
data_enc = pd.read_csv("../data/compas/compas_train_enc_1.csv")

X_train = data_enc[['age_cat.1', 'age_cat.2',
       'age_cat.3', 'sex.1', 'priors_count.1', 'priors_count.2',
       'priors_count.3', 'priors_count.4', 'c_charge_degree.1',
       'length_of_stay.1', 'length_of_stay.2', 'length_of_stay.3',
       'length_of_stay.4', 'length_of_stay.5']]
y_train = data_enc[['target']]
P = data[['race']]
L = data[['priors_count','c_charge_degree']]


fcl = FairTreeClassifier(depth = 1, time_limit = 10, _lambda = 0, positive_class = 1,
fairness_type = 'PE', fairness_bound = 1, num_threads = 1)

fcl.fit(X_train, y_train, P, L)
