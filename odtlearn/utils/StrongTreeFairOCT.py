"""
This module formulate the FairOCT problem in gurobipy.
"""

from gurobipy import Model, quicksum, LinExpr, GRB
import numpy as np
import pandas as pd
from itertools import combinations


class FairOCT:
    def __init__(
        self,
        X,
        y,
        tree,
        X_col_labels,
        labels,
        _lambda,
        time_limit,
        num_threads,
        fairness_type,
        fairness_bound,
        positive_class,
        P,
        P_col_labels,
        l,
        obj_mode,
        verbose,
    ):
        """
        :param X: numpy matrix or pandas data-frame of covariates.
                  It's up to the user to include the protected features in X or not.
                  We assume that we are allowed to branch on any of the columns within X.
        :param y: numpy array or pandas series/data-frame of class labels
        :param tree: Tree object
        :param X_col_labels: The column names of matrix X
        :param labels: the unique values of y
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        :param num_threads: Number of threads for the solver to use
        :param fairness_type: The type of the fairness constraint you wish to enforce, e.g., SP
        :param fairness_bound: The fairness bound,  a real value between (0,1]
        :param positive_class:
        :param P: P Is the np.array of the protected features. Its dimension is (n_sample, n_p) where n_p is number of
                  protected feaures.
        :param l: numpy array or pandas series/data-frame of legitimate feature
        :param P_col_labels: Names of the protected columns
        :param obj_mode: if obj_mode=acc we maximize the acc; if obj_mode = balance we maximize the balanced acc
        :param verbose: Display Gurobi model output
        """

        self.X = pd.DataFrame(X, columns=X_col_labels)
        self.y = y
        self.P = P
        self.l = l
        self.obj_mode = obj_mode

        self.class_name = "class_label"
        self.legitimate_name = "legitimate_feature_name"
        self.X_p = np.concatenate((P, l.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        self.X_p = pd.DataFrame(
            self.X_p,
            columns=(P_col_labels.tolist() + [self.legitimate_name, self.class_name]),
        )

        self.X_col_labels = X_col_labels
        self.labels = labels

        self.P_col_labels = P_col_labels

        # datapoints contains the indicies of our training data
        self.datapoints = np.arange(0, self.X.shape[0])

        self.tree = tree
        self._lambda = _lambda

        self.fairness_type = fairness_type
        self.fairness_bound = fairness_bound
        self.positive_class = positive_class

        # Decision Variables
        self.b = 0
        self.p = 0
        self.w = 0
        self.zeta = 0
        self.z = 0

        # Gurobi model
        self.model = Model("FairOCT")
        if not verbose:
            # supress all logging
            self.model.params.OutputFlag = 0
        if num_threads is not None:
            self.model.params.Threads = num_threads
        self.model.params.TimeLimit = time_limit

    def add_fairness_constraint(self, p_df, p_prime_df):
        count_p = p_df.shape[0]
        count_p_prime = p_prime_df.shape[0]
        constraint_added = False
        if count_p != 0 and count_p_prime != 0:
            constraint_added = True
            self.model.addConstr(
                (
                    (1 / count_p)
                    * quicksum(
                        quicksum(
                            self.zeta[i, n, self.positive_class]
                            for n in self.tree.Leaves + self.tree.Nodes
                        )
                        for i in p_df.index
                    )
                    - (
                        (1 / count_p_prime)
                        * quicksum(
                            quicksum(
                                self.zeta[i, n, self.positive_class]
                                for n in self.tree.Leaves + self.tree.Nodes
                            )
                            for i in p_prime_df.index
                        )
                    )
                )
                <= self.fairness_bound
            )

            self.model.addConstr(
                (
                    (1 / count_p)
                    * quicksum(
                        quicksum(
                            self.zeta[i, n, self.positive_class]
                            for n in (self.tree.Leaves + self.tree.Nodes)
                        )
                        for i in p_df.index
                    )
                )
                - (
                    (1 / count_p_prime)
                    * quicksum(
                        quicksum(
                            self.zeta[i, n, self.positive_class]
                            for n in self.tree.Leaves + self.tree.Nodes
                        )
                        for i in p_prime_df.index
                    )
                )
                >= -1 * self.fairness_bound
            )

        return constraint_added

    def create_primal_problem(self):
        """
        This function create and return a gurobi model formulating the FairOCT problem
        :return:  gurobi model object with the FairOCT formulation
        """

        ###########################################################
        # Define Variables
        ###########################################################

        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(
            self.tree.Nodes, self.X_col_labels, vtype=GRB.BINARY, name="b"
        )
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name="p"
        )
        # w[n,k]=1 iff at node n we predict class k
        self.w = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves,
            self.labels,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="w",
        )
        # zeta[i,n,k] is the amount of flow through the edge connecting node n to sink node t,k for datapoint i
        self.zeta = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            self.labels,
            vtype=GRB.BINARY,
            lb=0,
            name="zeta",
        )
        # z[i,n] is the incoming flow to node n for datapoint i to terminal node k
        self.z = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            vtype=GRB.BINARY,
            lb=0,
            name="z",
        )

        ###########################################################
        # Define Constraints
        ###########################################################

        # z[i,n] = z[i,l(n)] + z[i,r(n)] + (zeta[i,n,k] for all k in Labels)    forall i, n in Nodes
        for n in self.tree.Nodes:
            n_left = int(self.tree.get_left_children(n))
            n_right = int(self.tree.get_right_children(n))
            self.model.addConstrs(
                (
                    self.z[i, n]
                    == self.z[i, n_left]
                    + self.z[i, n_right]
                    + quicksum(self.zeta[i, n, k] for k in self.labels)
                )
                for i in self.datapoints
            )

        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs(
                self.z[i, int(self.tree.get_left_children(n))]
                <= quicksum(
                    self.b[n, f] for f in self.X_col_labels if self.X.at[i, f] == 0
                )
                for n in self.tree.Nodes
            )

        # z[i,r(n)] <= sum(b[n,f], f if x[i,f]=1)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs(
                (
                    self.z[i, int(self.tree.get_right_children(n))]
                    <= quicksum(
                        self.b[n, f] for f in self.X_col_labels if self.X.at[i, f] == 1
                    )
                )
                for n in self.tree.Nodes
            )

        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self.model.addConstrs(
            (
                quicksum(self.b[n, f] for f in self.X_col_labels)
                + self.p[n]
                + quicksum(self.p[m] for m in self.tree.get_ancestors(n))
                == 1
            )
            for n in self.tree.Nodes
        )

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self.model.addConstrs(
            (self.p[n] + quicksum(self.p[m] for m in self.tree.get_ancestors(n)) == 1)
            for n in self.tree.Leaves
        )

        # sum(w[n,k], k in labels) = p[n]
        for n in self.tree.Nodes + self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n, k] <= self.w[n, k]
                for i in self.datapoints
                for k in self.labels
            )

        # sum(w[n,k] for k in labels) == p[n]
        self.model.addConstrs(
            (quicksum(self.w[n, k] for k in self.labels) == self.p[n])
            for n in self.tree.Nodes + self.tree.Leaves
        )

        # z[i,n] == sum(zeta[i,n,k], k in labels)
        for n in self.tree.Leaves:
            self.model.addConstrs(
                quicksum(self.zeta[i, n, k] for k in self.labels) == self.z[i, n]
                for i in self.datapoints
            )

        # z[i,1] = 1 for all i datapoints
        self.model.addConstrs(self.z[i, 1] == 1 for i in self.datapoints)

        ###########################################################
        # Fairness Constraints
        ###########################################################

        # Loop through all possible combinations of the protected feature
        for protected_feature in self.P_col_labels:
            for combo in combinations(self.X_p[protected_feature].unique(), 2):
                p = combo[0]
                p_prime = combo[1]

                if self.fairness_type == "SP":
                    p_df = self.X_p[self.X_p[protected_feature] == p]
                    p_prime_df = self.X_p[self.X_p[protected_feature] == p_prime]
                    self.add_fairness_constraint(p_df, p_prime_df)
                elif self.fairness_type == "PE":
                    p_df = self.X_p[
                        (self.X_p[protected_feature] == p)
                        & (self.X_p[self.class_name] != self.positive_class)
                    ]
                    p_prime_df = self.X_p[
                        (self.X_p[protected_feature] == p_prime)
                        & (self.X_p[self.class_name] != self.positive_class)
                    ]
                    self.add_fairness_constraint(p_df, p_prime_df)
                elif self.fairness_type == "EOpp":
                    p_df = self.X_p[
                        (self.X_p[protected_feature] == p)
                        & (self.X_p[self.class_name] == self.positive_class)
                    ]
                    p_prime_df = self.X_p[
                        (self.X_p[protected_feature] == p_prime)
                        & (self.X_p[self.class_name] == self.positive_class)
                    ]
                    self.add_fairness_constraint(p_df, p_prime_df)
                elif (
                    self.fairness_type == "EOdds"
                ):  # Need to check with group if this is how we want to enforce this constraint
                    PE_p_df = self.X_p[
                        (self.X_p[protected_feature] == p)
                        & (self.X_p[self.class_name] != self.positive_class)
                    ]
                    PE_p_prime_df = self.X_p[
                        (self.X_p[protected_feature] == p_prime)
                        & (self.X_p[self.class_name] != self.positive_class)
                    ]

                    EOpp_p_df = self.X_p[
                        (self.X_p[protected_feature] == p)
                        & (self.X_p[self.class_name] == self.positive_class)
                    ]
                    EOpp_p_prime_df = self.X_p[
                        (self.X_p[protected_feature] == p_prime)
                        & (self.X_p[self.class_name] == self.positive_class)
                    ]

                    if (
                        PE_p_df.shape[0] != 0
                        and PE_p_prime_df.shape[0] != 0
                        and EOpp_p_df.shape[0] != 0
                        and EOpp_p_prime_df.shape[0] != 0
                    ):
                        self.add_fairness_constraint(PE_p_df, PE_p_prime_df)
                        self.add_fairness_constraint(EOpp_p_df, EOpp_p_prime_df)
                elif self.fairness_type == "CSP":
                    for l_value in self.X_p[self.legitimate_name].unique():
                        p_df = self.X_p[
                            (self.X_p[protected_feature] == p)
                            & (self.X_p[self.legitimate_name] == l_value)
                        ]
                        p_prime_df = self.X_p[
                            (self.X_p[protected_feature] == p_prime)
                            & (self.X_p[self.legitimate_name] == l_value)
                        ]
                        self.add_fairness_constraint(p_df, p_prime_df)
        ###########################################################
        # Define Objective
        ###########################################################
        # Max sum(sum(zeta[i,n,y(i)]))
        obj = LinExpr(0)
        for n in self.tree.Nodes:
            for f in self.X_col_labels:
                obj.add(-1 * self._lambda * self.b[n, f])
        if self.obj_mode == "acc":
            for i in self.datapoints:
                for n in self.tree.Nodes + self.tree.Leaves:
                    obj.add((1 - self._lambda) * (self.zeta[i, n, self.y[i]]))
        elif self.obj_mode == "balance":
            for i in self.datapoints:
                for n in self.tree.Nodes + self.tree.Leaves:
                    obj.add(
                        (1 - self._lambda)
                        * (
                            1
                            / self.y[self.y == self.y[i]].shape[0]
                            / self.labels.shape[0]
                        )
                        * (self.zeta[i, n, self.y[i]])
                    )
        else:
            assert self.obj_mode not in [
                "acc",
                "balance",
            ], f"Wrong objective mode. obj_mode should be one of acc or balance."

        self.model.setObjective(obj, GRB.MAXIMIZE)
