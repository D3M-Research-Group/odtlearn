from gurobipy import GRB, LinExpr, quicksum

from odtlearn.utils.prescriptive_formulation import PrescriptiveProblem


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

        self.b = self.model.addVars(
            self.tree.Nodes, self.X_col_labels, vtype=GRB.BINARY, name="b"
        )
        self.p = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name="p"
        )
        self.w = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves,
            self.treatments,
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

    def _define_constraints(self):
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
            (quicksum(self.w[n, k] for k in self.treatments) == self.p[n])
            for n in self.tree.Nodes + self.tree.Leaves
        )

        for n in self.tree.Leaves:
            self.model.addConstrs(
                self.zeta[i, n] == self.z[i, n] for i in self.datapoints
            )

    def _define_objective(self):
        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            obj.add(self.z[i, 1] * (self.y[i]) / self.ipw[i])

        self.model.setObjective(obj, GRB.MAXIMIZE)
