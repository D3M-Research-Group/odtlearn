from gurobipy import GRB, quicksum

from odtlearn.opt_ct import OptimalClassificationTree


class FlowOCTMultipleSink(OptimalClassificationTree):
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

    def _tree_struc_variables(self):
        # b[n,f] ==1 iff at node n we branch on feature f
        self._b = self._model.addVars(
            self._tree.Nodes, self._X_col_labels, vtype=GRB.BINARY, name="b"
        )
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self._p = self._model.addVars(
            self._tree.Nodes + self._tree.Leaves, vtype=GRB.BINARY, name="p"
        )

        # For classification w[n,k]=1 iff at node n we predict class k
        self._w = self._model.addVars(
            self._tree.Nodes + self._tree.Leaves,
            self._labels,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="w",
        )

    def _flow_variables(self):
        # zeta[i,n,k] is the amount of flow through the edge connecting node n to sink node t,k for datapoint i
        self._zeta = self._model.addVars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            self._labels,
            vtype=GRB.BINARY,
            lb=0,
            name="zeta",
        )
        # z[i,n] is the incoming flow to node n for datapoint i to terminal node k
        self._z = self._model.addVars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            vtype=GRB.BINARY,
            lb=0,
            name="z",
        )

    def _define_variables(self):
        self._tree_struc_variables()
        self._flow_variables()

    def _tree_structure_constraints(self):
        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self._model.addConstrs(
            (
                quicksum(self._b[n, f] for f in self._X_col_labels)
                + self._p[n]
                + quicksum(self._p[m] for m in self._tree.get_ancestors(n))
                == 1
            )
            for n in self._tree.Nodes
        )

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self._model.addConstrs(
            (
                self._p[n] + quicksum(self._p[m] for m in self._tree.get_ancestors(n))
                == 1
            )
            for n in self._tree.Leaves
        )

        # sum(w[n,k], k in labels) = p[n]
        for n in self._tree.Nodes + self._tree.Leaves:
            self._model.addConstrs(
                self._zeta[i, n, k] <= self._w[n, k]
                for i in self._datapoints
                for k in self._labels
            )

        # sum(w[n,k] for k in labels) == p[n]
        self._model.addConstrs(
            (quicksum(self._w[n, k] for k in self._labels) == self._p[n])
            for n in self._tree.Nodes + self._tree.Leaves
        )

    def _flow_constraints(self):
        # Flow Constraints
        # z[i,n] = z[i,l(n)] + z[i,r(n)] + (zeta[i,n,k] for all k in Labels)    forall i, n in Nodes
        for n in self._tree.Nodes:
            n_left = int(self._tree.get_left_children(n))
            n_right = int(self._tree.get_right_children(n))
            self._model.addConstrs(
                (
                    self._z[i, n]
                    == self._z[i, n_left]
                    + self._z[i, n_right]
                    + quicksum(self._zeta[i, n, k] for k in self._labels)
                )
                for i in self._datapoints
            )

        # z[i,n] == sum(zeta[i,n,k], k in labels)
        for n in self._tree.Leaves:
            self._model.addConstrs(
                quicksum(self._zeta[i, n, k] for k in self._labels) == self._z[i, n]
                for i in self._datapoints
            )

    def _arc_constraints(self):
        # Arc constraints
        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self._datapoints:
            self._model.addConstrs(
                self._z[i, int(self._tree.get_left_children(n))]
                <= quicksum(
                    self._b[n, f] for f in self._X_col_labels if self._X.at[i, f] == 0
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

        for n in self._tree.Nodes + self._tree.Leaves:
            self._model.addConstrs(
                self._zeta[i, n, k] <= self._w[n, k]
                for i in self._datapoints
                for k in self._labels
            )

        # z[i,1] = 1 for all i datapoints
        self._model.addConstrs(self._z[i, 1] == 1 for i in self._datapoints)

    def _define_constraints(self):
        self._tree_structure_constraints()
        self._flow_constraints()
        self._arc_constraints()
