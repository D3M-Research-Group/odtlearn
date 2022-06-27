from gurobipy import GRB, LinExpr, quicksum

from odtlearn.utils.problem_formulation import ProblemFormulation


class PrescriptiveTreeFormulation(ProblemFormulation):
    def __init__(
        self,
        X,
        t,
        y,
        ipw,
        treatments_set,
        tree,
        X_col_labels,
        model_name,
        time_limit,
        num_threads,
        verbose,
    ) -> None:

        self.t = t
        self.ipw = ipw
        self.treatments_set = treatments_set

        # Decision Variables
        self.b = 0
        self.p = 0
        self.w = 0
        self.zeta = 0
        self.z = 0

        super().__init__(
            X, y, tree, X_col_labels, model_name, time_limit, num_threads, verbose
        )


class FlowOPT_Robust(PrescriptiveTreeFormulation):
    def __init__(
        self,
        X,
        t,
        y,
        ipw,
        y_hat,
        robust,
        treatments_set,
        tree,
        X_col_labels,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        """
        :param X: numpy matrix or pandas dataframe of covariates
        :param t: numpy array or pandas series/dataframe of treatment assignments
        :param y: numpy array or pandas series/dataframe of observed outcomes
        :param ipw: numpy array or pandas series/dataframe of inverse propensity weights
        :param y_hat: numpy matrix or pandas dataframe of counterfactual estimates
        :param robust: Boolean indicating whether or not the FlowOPT method should be Doubly Robust (True)
                        or Direct Method (False)
        :param treatments_set: a list or set of all possible treatments
        :param tree: Tree object
        :param X_col_labels: a list of features in the covariate space X
        :param time_limit: The given time limit for solving the MIP
        :param num_threads: Number of threads for the solver to use
        """
        self.model_name = "FlowOPT"
        self.y_hat = y_hat
        self.robust = robust

        super().__init__(
            X,
            t,
            y,
            ipw,
            treatments_set,
            tree,
            X_col_labels,
            self.model_name,
            time_limit,
            num_threads,
            verbose,
        )

    def define_variables(self):
        self.b = self.model.addVars(
            self.tree.Nodes, self.X_col_labels, vtype=GRB.BINARY, name="b"
        )
        self.p = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name="p"
        )
        self.w = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves,
            self.treatments_set,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="w",
        )
        self.zeta = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            self.treatments_set,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="zeta",
        )
        self.z = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="z",
        )

    def define_constraints(self):
        # define constraints
        # z[i,n] = z[i,l(n)] + z[i,r(n)] + zeta[i,n]    forall i, n in Nodes
        for n in self.tree.Nodes:
            n_left = int(self.tree.get_left_children(n))
            n_right = int(self.tree.get_right_children(n))
            self.model.addConstrs(
                (
                    self.z[i, n]
                    == self.z[i, n_left]
                    + self.z[i, n_right]
                    + quicksum(self.zeta[i, n, k] for k in self.treatments_set)
                )
                for i in self.datapoints
            )

        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]<=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs(
                (
                    self.z[i, int(self.tree.get_left_children(n))]
                    <= quicksum(
                        self.b[n, f] for f in self.X_col_labels if self.X.at[i, f] <= 0
                    )
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

        # zeta[i,n] <= w[n,T[i]] for all n in N+L, i
        for n in self.tree.Nodes + self.tree.Leaves:
            for k in self.treatments_set:
                self.model.addConstrs(
                    self.zeta[i, n, k] <= self.w[n, k] for i in self.datapoints
                )

        # sum(w[n,k], k in treatments) = p[n]
        self.model.addConstrs(
            (quicksum(self.w[n, k] for k in self.treatments_set) == self.p[n])
            for n in self.tree.Nodes + self.tree.Leaves
        )

        for n in self.tree.Leaves:
            self.model.addConstrs(
                quicksum(self.zeta[i, n, k] for k in self.treatments_set)
                == self.z[i, n]
                for i in self.datapoints
            )

        self.model.addConstrs(self.z[i, 1] == 1 for i in self.datapoints)

    def define_objective(self):
        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            for n in self.tree.Nodes + self.tree.Leaves:
                for k in self.treatments_set:
                    obj.add(
                        self.zeta[i, n, k] * (self.y_hat[i][int(k)])
                    )  # we assume that each column corresponds to an ordered list t, which might be problematic
                    treat = self.t[i]
                    if self.robust:
                        if int(treat) == int(k):
                            obj.add(
                                self.zeta[i, n, k]
                                * (self.y[i] - self.y_hat[i][int(k)])
                                / self.ipw[i]
                            )

        self.model.setObjective(obj, GRB.MAXIMIZE)


class FlowOPT_IPW(PrescriptiveTreeFormulation):
    def __init__(
        self,
        X,
        t,
        y,
        ipw,
        treatments_set,
        tree,
        X_col_labels,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        self.model_name = "IPW"
        super().__init__(
            X,
            t,
            y,
            ipw,
            treatments_set,
            tree,
            X_col_labels,
            self.model_name,
            time_limit,
            num_threads,
            verbose,
        )

    def define_variables(self):
        # define variables

        self.b = self.model.addVars(
            self.tree.Nodes, self.X_col_labels, vtype=GRB.BINARY, name="b"
        )
        self.p = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name="p"
        )
        self.w = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves,
            self.treatments_set,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="w",
        )
        self.zeta = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="zeta",
        )
        self.z = self.model.addVars(
            self.datapoints,
            self.tree.Nodes + self.tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="z",
        )

    def define_constraints(self):
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

        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]<=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs(
                (
                    self.z[i, int(self.tree.get_left_children(n))]
                    <= quicksum(
                        self.b[n, f] for f in self.X_col_labels if self.X.at[i, f] <= 0
                    )
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

        # zeta[i,n] <= w[n,T[i]] for all n in N+L, i
        for n in self.tree.Nodes + self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n] <= self.w[n, self.t[i]] for i in self.datapoints
            )

        # sum(w[n,k], k in treatments) = p[n]
        self.model.addConstrs(
            (quicksum(self.w[n, k] for k in self.treatments_set) == self.p[n])
            for n in self.tree.Nodes + self.tree.Leaves
        )

        for n in self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n] == self.z[i, n] for i in self.datapoints
            )

    def define_objective(self):
        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            obj.add(self.z[i, 1] * (self.y[i]) / self.ipw[i])

        self.model.setObjective(obj, GRB.MAXIMIZE)
