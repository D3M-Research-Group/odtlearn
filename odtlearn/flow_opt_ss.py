from odtlearn import ODTL
from odtlearn.opt_pt import OptimalPrescriptiveTree


class FlowOPTSingleSink(OptimalPrescriptiveTree):
    """
    A class for learning optimal prescriptive trees with a single sink node using flow optimization.

    Parameters
    ----------
    solver : str
        The solver to use for the optimization problem. Can be either "gurobi" or "cbc".
    depth : int
        The maximum depth of the tree to be learned.
    time_limit : int
        The time limit (in seconds) for solving the optimization problem.
    num_threads : int
        The number of threads to use for solving the optimization problem.
    verbose : bool
        Whether to print verbose output during the tree learning process.

    Attributes
    ----------
    _b : dict
        A dictionary of binary decision variables representing the branching decisions at each node.
    _p : dict
        A dictionary of binary decision variables representing the prediction decisions at each node.
    _w : dict
        A dictionary of continuous decision variables representing the treatment weights at each node.
    _z : dict
        A dictionary of continuous decision variables representing the flow of each datapoint to each node.

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
    _define_constraints()
        Defines all the constraints used in the optimization problem.

    Notes
    -----
    This is a base class and should not be instantiated directly. Instead, use one of the
    derived classes that implement a specific prescriptive tree method, such as
    :mod:`FlowOPT_IPW <odtlearn.flow_opt_ipw.FlowOPT_IPW>`.

    This class extends the :mod:`OptimalPrescriptiveTree <odtlearn.opt_pt.OptimalPrescriptiveTree>` class
    to learn optimal prescriptive trees
    with a single sink node using flow optimization. It formulates the problem as a mixed-integer
    program (MIP) and solves it using either the Gurobi or CBC solver.

    The key idea is to model the flow of each datapoint through the tree, allowing it to reach
    a single sink node (i.e., leaf) with a specific treatment recommendation. The objective
    is to optimize the treatment recommendation based on the characteristics of each datapoint.

    The class defines decision variables and constraints specific to the flow optimization
    formulation with a single sink node.
    The :meth:`_define_variables <odtlearn.flow_opt_ss.FlowOPTSingleSink._define_variables>` method defines
    the decision variables, including the tree structure variables (`_b`, `_p`, `_w`) and the flow variable (`_z`).

    The :meth:`_define_constraints <odtlearn.flow_opt_ss.FlowOPTSingleSink._define_constraints>` method
    defines the constraints, including the tree structure
    constraints and flow constraints. These constraints ensure the validity of the tree structure
    and the proper flow of datapoints through the tree to a single sink node.

    The class inherits the :meth:`fit <odtlearn.opt_pt.OptimalPrescriptiveTree.fit>`,
    :meth:`predict <odtlearn.opt_pt.OptimalPrescriptiveTree.predict>`,
    :meth:`print_tree <odtlearn.opt_pt.OptimalPrescriptiveTree.print_tree>`,
    and :meth:`plot_tree <odtlearn.opt_pt.OptimalPrescriptiveTree.plot_tree>` methods from the
    :mod:`OptimalPrescriptiveTree <odtlearn.opt_pt.OptimalPrescriptiveTree>` class to learn the optimal prescriptive
    tree, make predictions,
    and visualize the learned tree.
    """

    def __init__(
        self,
        solver,
        depth,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        super().__init__(
            solver,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _tree_struc_variables(self):
        self._b = self._solver.add_vars(
            self._tree.Nodes, self._X_col_labels, vtype=ODTL.BINARY, name="b"
        )
        self._p = self._solver.add_vars(
            self._tree.Nodes + self._tree.Leaves, vtype=ODTL.BINARY, name="p"
        )
        self._w = self._solver.add_vars(
            self._tree.Nodes + self._tree.Leaves,
            self._treatments,
            vtype=ODTL.CONTINUOUS,
            lb=0,
            name="w",
        )

    def _flow_variables(self):
        self._zeta = self._solver.add_vars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            vtype=ODTL.CONTINUOUS,
            lb=0,
            name="zeta",
        )
        self._z = self._solver.add_vars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            vtype=ODTL.CONTINUOUS,
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

        # sum(w[n,k], k in treatments) = p[n]
        self._solver.add_constrs(
            (
                self._solver.quicksum(self._w[n, k] for k in self._treatments)
                == self._p[n]
            )
            for n in self._tree.Nodes + self._tree.Leaves
        )

    def _flow_constraints(self):
        # z[i,n] = z[i,l(n)] + z[i,r(n)] + zeta[i,n]    forall i, n in Nodes
        for n in self._tree.Nodes:
            n_left = int(self._tree.get_left_children(n))
            n_right = int(self._tree.get_right_children(n))
            self._solver.add_constrs(
                (
                    self._z[i, n]
                    == self._z[i, n_left] + self._z[i, n_right] + self._zeta[i, n]
                )
                for i in self._datapoints
            )

        for n in self._tree.Leaves:
            self._solver.add_constrs(
                self._zeta[i, n] == self._z[i, n] for i in self._datapoints
            )

    def _arc_constraints(self):
        # z[i,l(n)] <= sum(b[n,f], f if x[i,f]<=0)    forall i, n in Nodes
        for i in self._datapoints:
            self._solver.add_constrs(
                (
                    self._z[i, int(self._tree.get_left_children(n))]
                    <= self._solver.quicksum(
                        self._b[n, f]
                        for f in self._X_col_labels
                        if self._X.at[i, f] <= 0
                    )
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

        # zeta[i,n] <= w[n,T[i]] for all n in N+L, i
        for n in self._tree.Nodes + self._tree.Leaves:
            self._solver.add_constrs(
                self._zeta[i, n] <= self._w[n, self._t[i]] for i in self._datapoints
            )

    def _define_constraints(self):
        self._tree_structure_constraints()
        self._flow_constraints()
        self._arc_constraints()

    def _define_objective(self):
        # define objective function
        obj = self._solver.lin_expr(0)
        for i in self._datapoints:
            obj += self._z[i, 1] * (self._y[i]) / self._ipw[i]

        self._solver.set_objective(obj, ODTL.MAXIMIZE)
