from itertools import combinations

import numpy as np
import pandas as pd
from gurobipy import GRB, LinExpr, quicksum
from odtlearn.utils.problem_formulation import ProblemFormulation


class StrongTreeFormulation(ProblemFormulation):
    def __init__(
        self,
        X,
        y,
        tree,
        X_col_labels,
        labels,
        _lambda,
        obj_mode,
        model_name,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        self.model_name = model_name
        self._lambda = _lambda
        self.obj_mode = obj_mode
        self.labels = labels
        super().__init__(
            X, y, tree, X_col_labels, self.model_name, time_limit, num_threads, verbose
        )

        # Decision Variables
        if self.model_name in ["FlowOCT", "FairOCT"]:
            self.b = 0
            self.p = 0
            self.w = 0
            self.zeta = 0
            self.z = 0
        elif self.model_name in ["BendersOCT"]:
            self.g = 0
            self.b = 0
            self.p = 0
            self.w = 0
            # The cuts we add in the callback function would be treated as lazy constraints
            self.model.params.LazyConstraints = 1
            """
            The following variables are used for the Benders problem to keep track
            of the times we call the callback.

            - counter_integer tracks number of times we call the callback from an
            integer node in the branch-&-bound tree
                - time_integer tracks the associated time spent in the
                callback for these calls
            - counter_general tracks number of times we call the callback from
            a non-integer node in the branch-&-bound tree
                - time_general tracks the associated time spent in the callback for
                these calls

            the ones ending with success are related to success calls.
            By success we mean ending up adding a lazy constraint
            to the model

            """
            self.model._total_callback_time_integer = 0
            self.model._total_callback_time_integer_success = 0

            self.model._total_callback_time_general = 0
            self.model._total_callback_time_general_success = 0

            self.model._callback_counter_integer = 0
            self.model._callback_counter_integer_success = 0

            self.model._callback_counter_general = 0
            self.model._callback_counter_general_success = 0

            # We also pass the following information to the model as we need them in the callback
            self.model._main_grb_obj = self


class FlowOCT(StrongTreeFormulation):
    def __init__(
        self,
        X,
        y,
        tree,
        X_col_labels,
        labels,
        _lambda,
        obj_mode,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        """
        :param X: numpy matrix or pandas dataframe of covariates
        :param y: numpy array or pandas series/dataframe of class labels
        :param tree: Tree object
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        :param num_threads: Number of threads for the solver to use
        :param obj_mode: if obj_mode=acc we maximize the acc; if obj_mode = balance we maximize the balanced acc
        :param verbose: Display Gurobi model output
        """
        self.model_name = "FlowOCT"

        # initialize the StrongTreeFormulation
        super().__init__(
            X,
            y,
            tree,
            X_col_labels,
            labels,
            _lambda,
            obj_mode,
            self.model_name,
            time_limit,
            num_threads,
            verbose,
        )

    def create_primal_problem(self):
        """
        This function create and return a gurobi model formulating
        the FlowOCT problem
        :return:  gurobi model object with the FlowOCT formulation
        """

        ###########################################################
        # Define Variables
        ###########################################################

        # b[n,f] ==1 iff at node n we branch on feature f
        # do Gurobi variable names need to be strings?
        self.b = self.model.addVars(
            self.tree.Nodes, self.X_col_labels, vtype=GRB.BINARY, name="b"
        )
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name="p"
        )

        # For classification w[n,k]=1 iff at node n we predict class k
        self.w = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves,
            self.labels,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="w",
        )
        # zeta[i,n] is the amount of flow through the edge connecting node n
        # to sink node t for data-point i
        self.zeta = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="zeta",
        )
        # z[i,n] is the incoming flow to branching node n for data-point i
        self.z = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="z",
        )

        ###########################################################
        # Define Constraints
        ###########################################################

        # z[i,n] = z[i,l(n)] + z[i,r(n)] + zeta[i,n]    forall i, n in Nodes
        for n in self.tree.Nodes:
            n_left = int(self.tree.get_left_children(n))
            n_right = int(self.tree.get_right_children(n))
            self.model.addConstrs(
                (
                    self.z[i, n]
                    == self.z[i, n_left] + self.z[i, n_right] + self.zeta[i, n]
                )
                for i in self.datapoints
            )

        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]=0) forall i, n in Nodes
        # changed this to loop over the indicies of X and check if the column values at a given idx
        # equals zero
        for i in self.datapoints:
            self.model.addConstrs(
                (
                    self.z[i, int(self.tree.get_left_children(n))]
                    <= quicksum(
                        self.b[n, f] for f in self.X_col_labels if self.X.at[i, f] == 0
                    )
                )
                for n in self.tree.Nodes
            )

        # z[i,r(n)] <= sum(b[n,f], f if x[i,f]=1) forall i, n in Nodes
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

        # zeta[i,n] <= w[n,y[i]]     forall n in N+L, i
        for n in self.tree.Nodes + self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n] <= self.w[n, self.y[i]] for i in self.datapoints
            )

        # sum(w[n,k], k in labels) = p[n]
        self.model.addConstrs(
            (quicksum(self.w[n, k] for k in self.labels) == self.p[n])
            for n in self.tree.Nodes + self.tree.Leaves
        )

        for n in self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n] == self.z[i, n] for i in self.datapoints
            )

        ###########################################################
        # Define the Objective
        ###########################################################
        obj = LinExpr(0)
        for n in self.tree.Nodes:
            for f in self.X_col_labels:
                obj.add(-1 * self._lambda * self.b[n, f])
        if self.obj_mode == "acc":
            for i in self.datapoints:
                obj.add((1 - self._lambda) * self.z[i, 1])

        elif self.obj_mode == "balance":
            for i in self.datapoints:
                obj.add(
                    (1 - self._lambda)
                    * (1 / self.y[self.y == self.y[i]].shape[0] / self.labels.shape[0])
                    * self.z[i, 1]
                )
        else:
            assert self.obj_mode not in [
                "acc",
                "balance",
            ], "Wrong objective mode. obj_mode should be one of acc or balance."

        self.model.setObjective(obj, GRB.MAXIMIZE)


class BendersOCT(StrongTreeFormulation):
    def __init__(
        self,
        X,
        y,
        tree,
        X_col_labels,
        labels,
        _lambda,
        obj_mode,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        """
        :param X: numpy matrix of covariates
        :param y: numpy array of class labels
        :param tree: Tree object
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        :param verbose: Display Gurobi model output
        """
        self.model_name = "BendersOCT"
        super().__init__(
            X,
            y,
            tree,
            X_col_labels,
            labels,
            _lambda,
            obj_mode,
            self.model_name,
            time_limit,
            num_threads,
            verbose,
        )

    def create_main_problem(self):
        """
        This function create and return a gurobi model
        formulating the BendersOCT problem
        :return:  gurobi model object with the BendersOCT formulation
        """

        ###########################################################
        # Define Variables
        ###########################################################

        # g[i] is the objective value for the sub-problem[i]
        self.g = self.model.addVars(
            self.datapoints, vtype=GRB.CONTINUOUS, ub=1, name="g"
        )
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

        # we need these in the callback to have access to the value of the decision variables
        self.model._vars_g = self.g
        self.model._vars_b = self.b
        self.model._vars_p = self.p
        self.model._vars_w = self.w

        ###########################################################
        # Define Constraints
        ###########################################################

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

        # sum(w[n,k], k in labels) = p[n]
        self.model.addConstrs(
            (quicksum(self.w[n, k] for k in self.labels) == self.p[n])
            for n in self.tree.Nodes + self.tree.Leaves
        )

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self.model.addConstrs(
            (self.p[n] + quicksum(self.p[m] for m in self.tree.get_ancestors(n)) == 1)
            for n in self.tree.Leaves
        )

        ###########################################################
        # Define the Objective
        ###########################################################
        obj = LinExpr(0)
        for n in self.tree.Nodes:
            for f in self.X_col_labels:
                obj.add(-1 * self._lambda * self.b[n, f])
        if self.obj_mode == "acc":
            for i in self.datapoints:
                obj.add((1 - self._lambda) * self.g[i])
        elif self.obj_mode == "balance":
            for i in self.datapoints:
                obj.add(
                    (1 - self._lambda)
                    * (1 / self.y[self.y == self.y[i]].shape[0] / self.labels.shape[0])
                    * self.g[i]
                )
        else:
            assert self.obj_mode not in [
                "acc",
                "balance",
            ], "Wrong objective mode. obj_mode should be one of acc or balance."

        self.model.setObjective(obj, GRB.MAXIMIZE)


class FairOCT(StrongTreeFormulation):
    def __init__(
        self,
        X,
        y,
        tree,
        X_col_labels,
        labels,
        _lambda,
        fairness_type,
        fairness_bound,
        positive_class,
        protect_feat,
        protect_feat_col_labels,
        legit_factor,
        time_limit,
        num_threads,
        obj_mode,
        verbose,
    ) -> None:
        """
        :param X: numpy matrix or pandas data-frame of covariates.
                  It's up to the user to include the protected features in X or not.
                  We assume that we are allowed to branch on any of the columns within X.
        :param y: numpy array or pandas series/data-frame of class labels
        :param tree: Tree object
        :param X_col_labels: The column names of matrix X
        :param labels: the unique values of y
        :param _lambda: The regularization parameter in the objective
        :param fairness_type: The type of the fairness constraint you wish to enforce, e.g., SP
        :param fairness_bound: The fairness bound,  a real value between (0,1]
        :param positive_class:
        :param protect_feat: protect_feat Is the np.array of the protected features.
            Its dimension is (n_sample, n_p) where n_p is number of protected feaures.
        :param protect_feat_col_labels: Names of the protected columns
        :param legit_factor: numpy array or pandas series/data-frame of legitimate feature
        :param time_limit: The given time limit for solving the MIP
        :param num_threads: Number of threads for the solver to use
        :param obj_mode: if obj_mode=acc we maximize the acc;
        if obj_mode = balance we maximize the balanced acc
        :param verbose: Display Gurobi model output
        """
        self.model_name = "FairOCT"
        super().__init__(
            X,
            y,
            tree,
            X_col_labels,
            labels,
            _lambda,
            obj_mode,
            self.model_name,
            time_limit,
            num_threads,
            verbose,
        )
        self.P = protect_feat
        self.legit_factor = legit_factor

        self.class_name = "class_label"
        self.legitimate_name = "legitimate_feature_name"
        self.X_p = np.concatenate(
            (protect_feat, legit_factor.reshape(-1, 1), y.reshape(-1, 1)), axis=1
        )
        self.X_p = pd.DataFrame(
            self.X_p,
            columns=(
                protect_feat_col_labels.tolist()
                + [self.legitimate_name, self.class_name]
            ),
        )

        self.P_col_labels = protect_feat_col_labels

        self.fairness_type = fairness_type
        self.fairness_bound = fairness_bound
        self.positive_class = positive_class

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
            ], "Wrong objective mode. obj_mode should be one of acc or balance."

        self.model.setObjective(obj, GRB.MAXIMIZE)
