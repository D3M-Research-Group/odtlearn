"""
This module formulate the FlowOCT problem in gurobipy.
"""
from gurobipy import Model, GRB, quicksum, LinExpr
import numpy as np
import pandas as pd


class FlowOCT:
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
        obj_mode,
        verbose,
    ):
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
        self.X = pd.DataFrame(X, columns=X_col_labels)
        self.y = y

        self.X_col_labels = X_col_labels
        self.labels = labels
        # datapoints contains the indicies of our training data
        self.datapoints = np.arange(0, self.X.shape[0])

        self.tree = tree
        self._lambda = _lambda
        self.obj_mode = obj_mode

        # Decision Variables
        self.b = 0
        self.p = 0
        self.w = 0
        self.zeta = 0
        self.z = 0

        # Gurobi model
        self.model = Model("FlowOCT")
        if not verbose:
            # supress all logging
            self.model.params.OutputFlag = 0
        if num_threads is not None:
            self.model.params.Threads = num_threads
        self.model.params.TimeLimit = time_limit

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
        """
        For classification w[n,k]=1 iff at node n we predict class k
        """
        self.w = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves,
            self.labels,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="w",
        )
        """
        zeta[i,n] is the amount of flow through the edge connecting node n
        to sink node t for data-point i
        """
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
