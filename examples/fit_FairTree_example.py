import pandas as pd
from sklearn.model_selection import train_test_split
from trees.FairTree import FairTreeClassifier


data = pd.read_csv("./data/compas/compas_train_1.csv")
data_enc = pd.read_csv("./data/compas/compas_train_enc_1.csv")

X_train = data_enc[['race.1', 'race.2', 'race.3', 'race.4','age_cat.1', 'age_cat.2',
       'age_cat.3', 'sex.1', 'priors_count.1', 'priors_count.2',
       'priors_count.3', 'priors_count.4', 'c_charge_degree.1',
       'length_of_stay.1', 'length_of_stay.2', 'length_of_stay.3',
       'length_of_stay.4', 'length_of_stay.5']]
y_train = data_enc[['target']]
P = data[['race', 'sex']] # P could have multiple columns or only one
l = data[['priors_count']] # For now we assume that L has only a single column 


fcl = FairTreeClassifier(positive_class = 1, depth = 1, _lambda = 0, time_limit = 10,
        fairness_type = 'CSP', fairness_bound = 1, num_threads = 1)

fcl.fit(X_train, y_train, P, l)
