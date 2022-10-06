from gurobipy import GRB, quicksum

from odtlearn.opt_dt import OptimalDecisionTree


class FlowMultipleSink(OptimalDecisionTree):
    def __init__(
        self,
        depth,
        time_limit,
        num_threads,
        verbose,
        _lambda=None,
    ) -> None:

        self._lambda = _lambda

        super().__init__(
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _flow_vars(self):
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
        self._tree_struc_vars()
        self._flow_vars()

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
