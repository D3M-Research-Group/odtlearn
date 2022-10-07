# flake8: noqa
import matplotlib.pyplot as plt

from odtlearn.datasets import fairness_example, flow_oct_example
from odtlearn.BendersOCT import BendersOCT
from odtlearn.FairOCT import FairOCT

# Learn an optimal classification tree
X, y = flow_oct_example()
stcl = BendersOCT(depth=2, _lambda=0.01, obj_mode="acc")
stcl.fit(X, y)
stcl_predictions = stcl.predict(X)

fig, ax = plt.subplots(figsize=(5, 2.5))
stcl.plot_tree()
# plt.savefig('strong_tree.png', dpi=800, bbox_inches='tight')
plt.show()


# Learn an optimal fair classification tree
X, y, protect_feat, legit_factor = fairness_example()
fcl = FairOCT(
    positive_class=1,
    depth=2,
    _lambda=0.01,
    time_limit=100,
    fairness_type="SP",
    fairness_bound=1,
    obj_mode="acc",
)
fcl.fit(X, y, protect_feat, legit_factor)
fcl_predictions = fcl.predict(X)

fcl.fairness_metric_summary("SP")
fig, ax = plt.subplots(figsize=(5, 2.5))
fcl.plot_tree()
# plt.savefig('fair_tree.png', dpi=800, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
from odtlearn.datasets import robustness_example

# from odtlearn.RobustTree import RobustTreeClassifier
from odtlearn.RobustOCT import RobustOCT

X, y, costs = robustness_example()

# Learn an optimal classification tree robust to distribution shift
rbcl = RobustOCT(depth=2, time_limit=60)
rbcl.fit(X, y)
rbcl_predictions = rbcl.predict(X)

fig, ax = plt.subplots(figsize=(5, 2.5))
rbcl.plot_tree()
# plt.savefig('robust_tree.png', dpi=800, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
from odtlearn.datasets import prescriptive_ex_data

# from odtlearn.PrescriptiveTree import PrescriptiveTreeClassifier
from odtlearn.FlowOPT import FlowOPT_DR

df_train, df_test = prescriptive_ex_data()

X_train = df_train[["X1", "X2"]]
y_train, t_train, ipw_train = df_train[["y", "t", "prop_scores_logit"]].T.to_numpy()
potential_y_train = df_train[["y0", "y1"]]
y_hat_train = df_train[["linear0", "linear1"]]

# Learn an optimal prescriptive tree
pt = FlowOPT_DR(depth=2, time_limit=3600)
pt.fit(X=X_train, t=t_train, y=y_train, ipw=ipw_train, y_hat=y_hat_train)
pt_predictions = pt.predict(X)

fig, ax = plt.subplots(figsize=(5, 2.5))
pt.plot_tree()
# plt.savefig('prescriptive_tree.png', dpi=800, bbox_inches='tight')
plt.show()
