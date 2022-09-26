from gurobipy import GRB, quicksum

from odtlearn.utils.classification_formulation import ClassificationProblem


class FlowOCTMultipleNode(ClassificationProblem):
    def __init__(
        self,
        _lambda,
        depth,
        time_limit,
        num_threads,
        verbose,
    ) -> None:

        self._lambda = _lambda

        super().__init__(
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_variables(self):
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

    def _define_constraints(self):
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
