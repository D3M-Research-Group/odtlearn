from gurobipy import GRB, LinExpr, quicksum

from odtlearn.prescriptive_formulation import PrescriptiveProblem


class FlowOPTSingleNode(PrescriptiveProblem):
    def __init__(
        self,
        depth,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        super().__init__(
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_variables(self):
        # define variables

        self._b = self._model.addVars(
            self._tree.Nodes, self._X_col_labels, vtype=GRB.BINARY, name="b"
        )
        self.p = self._model.addVars(
            self._tree.Nodes + self._tree.Leaves, vtype=GRB.BINARY, name="p"
        )
        self._w = self._model.addVars(
            self._tree.Nodes + self._tree.Leaves,
            self._treatments,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="w",
        )
        self._zeta = self._model.addVars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="zeta",
        )
        self._z = self._model.addVars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="z",
        )

    def _define_constraints(self):
        # define constraints

        # z[i,n] = z[i,l(n)] + z[i,r(n)] + zeta[i,n]    forall i, n in Nodes
        for n in self._tree.Nodes:
            n_left = int(self._tree.get_left_children(n))
            n_right = int(self._tree.get_right_children(n))
            self._model.addConstrs(
                (
                    self._z[i, n]
                    == self._z[i, n_left] + self._z[i, n_right] + self._zeta[i, n]
                )
                for i in self._datapoints
            )

        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]<=0)    forall i, n in Nodes
        for i in self._datapoints:
            self._model.addConstrs(
                (
                    self._z[i, int(self._tree.get_left_children(n))]
                    <= quicksum(
                        self._b[n, f]
                        for f in self._X_col_labels
                        if self._X.at[i, f] <= 0
                    )
                )
                for n in self._tree.Nodes
            )

        # z[i,r(n)] <= sum(b[n,f], f if x[i,f]=1)    forall i, n in Nodes
        for i in self._datapoints:
            self._model.addConstrs(
                (
                    self._z[i, int(self._tree.get_right_children(n))]
                    <= quicksum(
                        self._b[n, f]
                        for f in self._X_col_labels
                        if self._X.at[i, f] == 1
                    )
                )
                for n in self._tree.Nodes
            )

        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self._model.addConstrs(
            (
                quicksum(self._b[n, f] for f in self._X_col_labels)
                + self.p[n]
                + quicksum(self.p[m] for m in self._tree.get_ancestors(n))
                == 1
            )
            for n in self._tree.Nodes
        )

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self._model.addConstrs(
            (self.p[n] + quicksum(self.p[m] for m in self._tree.get_ancestors(n)) == 1)
            for n in self._tree.Leaves
        )

        # zeta[i,n] <= w[n,T[i]] for all n in N+L, i
        for n in self._tree.Nodes + self._tree.Leaves:
            self._model.addConstrs(
                self._zeta[i, n] <= self._w[n, self._t[i]] for i in self._datapoints
            )

        # sum(w[n,k], k in treatments) = p[n]
        self._model.addConstrs(
            (quicksum(self._w[n, k] for k in self._treatments) == self.p[n])
            for n in self._tree.Nodes + self._tree.Leaves
        )

        for n in self._tree.Leaves:
            self._model.addConstrs(
                self._zeta[i, n] == self._z[i, n] for i in self._datapoints
            )

    def _define_objective(self):
        # define objective function
        obj = LinExpr(0)
        for i in self._datapoints:
            obj.add(self._z[i, 1] * (self._y[i]) / self.ipw[i])

        self._model.setObjective(obj, GRB.MAXIMIZE)
