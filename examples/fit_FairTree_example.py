"""
============================
FairTree Fit Example
============================

An example of fitting a FairTree decision tree using :class:`trees.FairTree.FairTreeClassifier`
"""
import pandas as pd
from trees.FairTree import FairTreeClassifier


data_train = pd.read_csv("./data/compas/compas_train_1.csv")
dat_train_enc = pd.read_csv("./data/compas/compas_train_enc_1.csv") #This is one-hot encoded version of data_train where every column is binary

data_test = pd.read_csv("./data/compas/compas_test_1.csv")
dat_test_enc = pd.read_csv("./data/compas/compas_test_enc_1.csv")

branching_features = [
        "race.1",
        "race.2",
        "race.3",
        "race.4",
        "age_cat.1",
        "age_cat.2",
        "age_cat.3",
        "sex.1",
        "priors_count.1",
        "priors_count.2",
        "priors_count.3",
        "priors_count.4",
        "c_charge_degree.1",
        "length_of_stay.1",
        "length_of_stay.2",
        "length_of_stay.3",
        "length_of_stay.4",
        "length_of_stay.5",
    ]

X_train = dat_train_enc[ branching_features]
y_train = dat_train_enc[["target"]]
P_train = data_train[["race", "sex"]]  # P could have multiple columns or only one
l_train = data_train[["priors_count"]]  # For now we assume that L has only a single column

X_test = dat_test_enc[ branching_features]
y_test = dat_test_enc[["target"]]
P_test = data_test[["race", "sex"]]  
l_test = data_test[["priors_count"]] 


fcl = FairTreeClassifier(
    positive_class=1,
    depth=1,
    _lambda=0,
    time_limit=10,
    fairness_type="CSP",
    fairness_bound=1,
    num_threads=1,
)

fcl_obj = fcl.fit(X_train, y_train, P_train, l_train)
pred_test = fcl.predict(X_test)
sp_val = fcl_obj.get_SP(P_test, y_test)
csp_val = fcl_obj.get_CSP(P_test, l_test, y_test)
eq_val = fcl_obj.get_EqOdds(P_test, y_test, pred_test)
ceq_val = fcl_obj.get_CondEqOdds(P_test, l_test, y_test, pred_test)


