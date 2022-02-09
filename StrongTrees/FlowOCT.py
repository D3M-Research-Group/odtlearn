"""
This module formulate the FlowOCT problem in gurobipy.
"""
from gurobipy import Model, GRB, quicksum, LinExpr
import numpy as np


class FlowOCT:
    def __init__(
        self, X, y, tree, X_col_labels, labels, _lambda, time_limit, mode, num_threads
    ):
        """
        :param X: numpy matrix or pandas dataframe of covariates
        :param y: numpy array or pandas series/dataframe of class labels
        :param tree: Tree object
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        :param mode: Regression vs Classification
        :param num_threads: Number of threads for the solver to use
        """
        self.mode = mode
        self.X = X
        self.y = y

        self.X_col_labels = X_col_labels
        self.labels = labels
        # datapoints contains the indicies of our training data
        self.datapoints = np.arange(0, self.X.shape[0])

        self.tree = tree
        self._lambda = _lambda

        # parameters
        self.m = {}
        for i in self.datapoints:
            self.m[i] = 1

        if self.mode == "regression":
            for i in self.datapoints:
                y_i = y[i]
                self.m[i] = max(y_i, 1 - y_i)

        # Decision Variables
        self.b = 0
        self.p = 0
        self.beta = 0
        self.zeta = 0
        self.z = 0

        # Gurobi model
        self.model = Model("FlowOCT")
        self.model.params.Threads = num_threads
        self.model.params.TimeLimit = time_limit

        """
        The following variables are used for the Benders problem to keep track
        of the times we call the callback.
        They are not used for this formulation.
        """
        self.model._total_callback_time_integer = 0
        self.model._total_callback_time_integer_success = 0

        self.model._total_callback_time_general = 0
        self.model._total_callback_time_general_success = 0

        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

        self.model._callback_counter_general = 0
        self.model._callback_counter_general_success = 0

    ###########################################################
    # Create the MIP formulation
    ###########################################################
    def create_master_problem(self):
        """
        This function create and return a gurobi model formulating
        the FlowOCT problem
        :return:  gurobi model object with the FlowOCT formulation
        """
        # define variables
        # b[n,f] ==1 iff at node n we branch on feature f
        # do Gurobi variable names need to be strings?
        self.b = self.model.addVars(
            self.tree.Nodes, self.X_col_labels, vtype=GRB.BINARY, name="b"
        )
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name="p"
        )
        """
        For classification beta[n,k]=1 iff at node n we predict class k
        For the case regression beta[n,1] is the prediction value for node n
        """
        self.beta = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves,
            self.labels,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="beta",
        )
        # zeta[i,n] is the amount of flow through the edge connecting node n
        #  to sink node t for datapoint i
        self.zeta = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="zeta",
        )
        # z[i,n] is the incoming flow to node n for datapoint i
        self.z = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="z",
        )

        # define constraints

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

        # z[i,l(n)] <= m[i] * sum(b[n,f], f if x[i,f]=0)
        # forall i, n in Nodes
        # changed this to loop over the indicies of X and check if the column values at a given idx
        # equals zero
        for i in self.datapoints:
            self.model.addConstrs(
                (
                    self.z[i, int(self.tree.get_left_children(n))]
                    <= self.m[i]
                    * quicksum(
                        self.b[n, f] for f in self.X_col_labels if self.X[i, f] == 0
                    )
                )
                for n in self.tree.Nodes
            )

        # z[i,r(n)] <= m[i] * sum(b[n,f], f if x[i,f]=1)
        # forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs(
                (
                    self.z[i, int(self.tree.get_right_children(n))]
                    <= self.m[i]
                    * quicksum(
                        self.b[n, f] for f in self.X_col_labels if self.X[i, f] == 1
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

        # sum(sum(b[n,f], f), n) <= branching_limit
        # self.model.addConstr(
        #     (quicksum(
        #         quicksum(self.b[n, f] for f in self.cat_features) for n in self.tree.Nodes)) <= self.branching_limit)

        # loss reduction:
        # zeta[i,n] <= beta[n,y[i]]     forall n in N+L, i
        if self.mode == "classification":
            # sum(beta[n,k], k in labels) = p[n]
            for n in self.tree.Nodes + self.tree.Leaves:
                self.model.addConstrs(
                    self.zeta[i, n] <= self.beta[n, self.y[i]] for i in self.datapoints
                )

            self.model.addConstrs(
                (quicksum(self.beta[n, k] for k in self.labels) == self.p[n])
                for n in self.tree.Nodes + self.tree.Leaves
            )

        elif self.mode == "regression":
            # beta[n,k] = 1
            for n in self.tree.Nodes + self.tree.Leaves:
                self.model.addConstrs(
                    self.zeta[i, n]
                    <= self.m[i] * self.p[n] - self.y[i] * self.p[n] + self.beta[n, 1]
                    for i in self.datapoints
                )

                self.model.addConstrs(
                    self.zeta[i, n]
                    <= self.m[i] * self.p[n] + self.y[i] * self.p[n] - self.beta[n, 1]
                    for i in self.datapoints
                )

            self.model.addConstrs(
                (self.beta[n, 1] <= self.p[n])
                for n in self.tree.Nodes + self.tree.Leaves
            )

        for n in self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n] == self.z[i, n] for i in self.datapoints
            )

        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            obj.add((1 - self._lambda) * (self.z[i, 1] - self.m[i]))

        for n in self.tree.Nodes:
            for f in self.cat_features:
                obj.add(-1 * self._lambda * self.b[n, f])

        self.model.setObjective(obj, GRB.MAXIMIZE)
