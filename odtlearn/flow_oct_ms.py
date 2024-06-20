from odtlearn import ODTL
from odtlearn.opt_ct import OptimalClassificationTree


class FlowOCTMultipleSink(OptimalClassificationTree):
    """
    A base class for learning optimal classification trees using flow-based formulation with multiple sink nodes.

    Parameters
    ----------
    solver : str
        The solver to use for the MIP formulation. Currently, only "gurobi" and "CBC" are supported.
    _lambda : float
        The regularization parameter for controlling the complexity of the learned tree.
    depth : int
        The maximum depth of the tree to be learned.
    time_limit : int
        The time limit (in seconds) for solving the MIP formulation.
    num_threads : int, optional
        The number of threads the solver should use. If not specified,
        solver uses all available threads
    verbose : bool, default=False
        Whether to print verbose output during the tree learning process.

    Attributes
    ----------
    _b : dict
        A dictionary of binary decision variables representing the branching decisions at each node.
    _p : dict
        A dictionary of binary decision variables representing the prediction decisions at each node.
    _w : dict
        A dictionary of continuous decision variables representing the prediction weights at each node.
    _zeta : dict
        A dictionary of binary decision variables representing the flow of each datapoint to each label at each node.
    _z : dict
        A dictionary of binary decision variables representing the flow of each datapoint to each node.

    Methods
    -------
    _tree_struc_variables()
        Defines the decision variables related to the tree structure.
    _flow_variables()
        Defines the decision variables related to the flow of datapoints.
    _define_variables()
        Defines all the decision variables used in the optimization problem.
    _tree_structure_constraints()
        Defines the constraints related to the tree structure.
    _flow_constraints()
        Defines the constraints related to the flow of datapoints.
    _arc_constraints()
        Defines the constraints related to the arcs between nodes.
    _define_constraints()
        Defines all the constraints used in the optimization problem.

    Notes
    -----
    This class extends the :mod:`OptimalClassificationTree <odtlearn.opt_dt.OptimalClassificationTree>` class
    and provides the basic structure and
    common functionality for flow-based optimal classification tree learning with multiple sink nodes.

    The key idea behind the flow-based formulation with multiple sink nodes is to model the flow of
    each datapoint through the tree, allowing it to reach multiple sink nodes (i.e., leaves) with
    different class predictions. The objective is to optimize the class predictions based on the
    characteristics of each datapoint.

    The class defines decision variables and constraints specific to the flow-based formulation
    with multiple sink nodes. The
    :meth:`_define_variables <odtlearn.flow_oct_ms.FlowOCTMultipleSink._define_variables>` method
    defines the decision variables,
    including the tree structure variables (`_b`, `_p`, `_w`) and the flow variables (`_zeta`, `_z`).

    The :meth:`_define_constraints <odtlearn.flow_oct_ms.FlowOCTMultipleSink._define_constraints>`
    method defines the constraints, including the tree structure
    constraints, flow constraints, and arc constraints. These constraints ensure the validity
    of the tree structure and the proper flow of datapoints through the tree to multiple sink nodes.

    Users should not instantiate this class directly, but instead use one of its derived classes
    such as :mod:`FlowOCT <odtlearn.flow_oct.FlowOCT>` for learning optimal classification trees.
    """

    def __init__(
        self,
        solver,
        _lambda,
        depth,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        self._lambda = _lambda

        super().__init__(
            solver,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _tree_struc_variables(self):
        # b[n,f] ==1 iff at node n we branch on feature f
        self._b = self._solver.add_vars(
            self._tree.Nodes, self._X_col_labels, vtype=ODTL.BINARY, name="b"
        )

        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self._p = self._solver.add_vars(
            self._tree.Nodes + self._tree.Leaves, vtype=ODTL.BINARY, name="p"
        )

        # For classification w[n,k]=1 iff at node n we predict class k
        self._w = self._solver.add_vars(
            self._tree.Nodes + self._tree.Leaves,
            self._labels,
            vtype=ODTL.CONTINUOUS,
            lb=0,
            name="w",
        )

    def _flow_variables(self):
        # zeta[i,n,k] is the amount of flow through the edge connecting node n to sink node t,k for datapoint i
        self._zeta = self._solver.add_vars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            self._labels,
            vtype=ODTL.BINARY,
            lb=0,
            name="zeta",
        )
        # z[i,n] is the incoming flow to node n for datapoint i to terminal node k
        self._z = self._solver.add_vars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            vtype=ODTL.BINARY,
            lb=0,
            name="z",
        )

    def _define_variables(self):
        self._tree_struc_variables()
        self._flow_variables()

    def _tree_structure_constraints(self):
        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self._solver.add_constrs(
            (
                self._solver.quicksum(self._b[n, f] for f in self._X_col_labels)
                + self._p[n]
                + self._solver.quicksum(self._p[m] for m in self._tree.get_ancestors(n))
                == 1
            )
            for n in self._tree.Nodes
        )

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self._solver.add_constrs(
            (
                self._p[n]
                + self._solver.quicksum(self._p[m] for m in self._tree.get_ancestors(n))
                == 1
            )
            for n in self._tree.Leaves
        )

        # sum(w[n,k], k in labels) = p[n]
        for n in self._tree.Nodes + self._tree.Leaves:
            self._solver.add_constrs(
                self._zeta[i, n, k] <= self._w[n, k]
                for i in self._datapoints
                for k in self._labels
            )

        # sum(w[n,k] for k in labels) == p[n]
        self._solver.add_constrs(
            (self._solver.quicksum(self._w[n, k] for k in self._labels) == self._p[n])
            for n in self._tree.Nodes + self._tree.Leaves
        )

    def _flow_constraints(self):
        # Flow Constraints
        # z[i,n] = z[i,l(n)] + z[i,r(n)] + (zeta[i,n,k] for all k in Labels)    forall i, n in Nodes
        for n in self._tree.Nodes:
            n_left = int(self._tree.get_left_children(n))
            n_right = int(self._tree.get_right_children(n))
            self._solver.add_constrs(
                (
                    self._z[i, n]
                    == self._z[i, n_left]
                    + self._z[i, n_right]
                    + self._solver.quicksum(self._zeta[i, n, k] for k in self._labels)
                )
                for i in self._datapoints
            )

        # z[i,n] == sum(zeta[i,n,k], k in labels)
        for n in self._tree.Leaves:
            self._solver.add_constrs(
                self._solver.quicksum(self._zeta[i, n, k] for k in self._labels)
                == self._z[i, n]
                for i in self._datapoints
            )

    def _arc_constraints(self):
        # Arc constraints
        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self._datapoints:
            self._solver.add_constrs(
                self._z[i, int(self._tree.get_left_children(n))]
                <= self._solver.quicksum(
                    self._b[n, f] for f in self._X_col_labels if self._X.at[i, f] == 0
                )
                for n in self._tree.Nodes
            )

        # z[i,r(n)] <= sum(b[n,f], f if x[i,f]=1)    forall i, n in Nodes
        for i in self._datapoints:
            self._solver.add_constrs(
                (
                    self._z[i, int(self._tree.get_right_children(n))]
                    <= self._solver.quicksum(
                        self._b[n, f]
                        for f in self._X_col_labels
                        if self._X.at[i, f] == 1
                    )
                )
                for n in self._tree.Nodes
            )

        for n in self._tree.Nodes + self._tree.Leaves:
            self._solver.add_constrs(
                self._zeta[i, n, k] <= self._w[n, k]
                for i in self._datapoints
                for k in self._labels
            )

        # z[i,1] = 1 for all i datapoints
        self._solver.add_constrs(self._z[i, 1] == 1 for i in self._datapoints)

    def _define_constraints(self):
        self._tree_structure_constraints()
        self._flow_constraints()
        self._arc_constraints()
