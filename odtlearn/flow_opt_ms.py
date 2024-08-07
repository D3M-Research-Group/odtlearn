from odtlearn import ODTL
from odtlearn.opt_pt import OptimalPrescriptiveTree


class FlowOPTMultipleSink(OptimalPrescriptiveTree):
    """
     A base class for learning optimal prescriptive trees with multiple sink nodes using flow optimization.

     Parameters
     ----------
     solver : str
         The solver to use for the optimization problem. Can be either "gurobi" or "cbc".
     depth : int, default=1
         The maximum depth of the tree to be learned.
     time_limit : int, default=60
         The time limit (in seconds) for solving the optimization problem.
     num_threads : int, default=None
         The number of threads the solver should use. If not specified,
         the solver uses all available threads.
     verbose : bool, default=False
         Whether to print verbose output during the tree learning process.

     Attributes
     ----------
     _b : dict
         A dictionary of binary decision variables representing the branching decisions at each node.
     _p : dict
         A dictionary of binary decision variables representing the prediction decisions at each node.
     _w : dict
         A dictionary of continuous decision variables representing the treatment weights at each node.
     _zeta : dict
         A dictionary of continuous decision variables representing the flow of each datapoint
         to each treatment at each node.
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
     _arc_constraints()
         Defines the constraints related to the arcs between nodes.
     _define_constraints()
         Defines all the constraints used in the optimization problem.
     _define_objective()
         Abstract method to be implemented by subclasses to define the objective function.
     fit(X, t, y, **kwargs)
         Abstract method to be implemented by subclasses to fit the optimal prescriptive tree.
    predict(X)
         Make treatment recommendations for the given input samples.

     Notes
     -----
     This is a base class for learning optimal prescriptive trees with multiple sink nodes using
     flow optimization. It extends the :mod:`OptimalPrescriptiveTree <odtlearn.opt_pt.OptimalPrescriptiveTree>` class
     and provides the basic
     structure and common functionality for flow-based prescriptive tree learning. It should not
     be instantiated directly. Instead, use one of the
     derived classes that implement a specific prescriptive tree method, such as
     :mod:`FlowOPT_DM <odtlearn.flow_opt.FlowOPT_DM>` or
     :mod:`FlowOPT_DR <odtlearn.flow_opt.FlowOPT_DR>`.

     The key idea is to model the flow of each datapoint through the tree, allowing it to reach
     multiple sink nodes (i.e., leaves) with different treatment recommendations. The objective
     is to optimize the treatment recommendations based on the characteristics of each datapoint.

     The class defines decision variables and constraints specific to the flow optimization
     formulation with multiple sink nodes.
     The :meth:`_define_variables <odtlearn.flow_opt_ms.FlowOPTMultipleSink._define_variables>` method defines
     the decision
     variables, including the tree structure variables (`_b`, `_p`, `_w`) and the flow variables
     (`_zeta`, `_z`).

     The :meth:`_define_constraints <odtlearn.flow_opt_ms.FlowOPTMultipleSink._define_constraints>` method defines
     the constraints,
     including the tree structure
     constraints, flow constraints, and arc constraints. These constraints ensure the validity
     of the tree structure and the proper flow of datapoints through the tree to multiple sink nodes.

     Subclasses of `FlowOPTMultipleSink` should implement the
     :meth:`_define_objective <odtlearn.flow_opt_ms.FlowOPTMultipleSink._define_objective>` method to specify
     the objective function for the optimization problem,
     and the :meth:`fit <odtlearn.flow_opt_ms.FlowOPTMultipleSink.fit>` method
     to handle the model
     fitting process.

     The class inherits the :meth:`predict <odtlearn.opt_pt.OptimalPrescriptiveTree.predict>` method from the
     :mod:`OptimalPrescriptiveTree <odtlearn.opt_pt.OptimalPrescriptiveTree>` class to make
     treatment recommendations based on the learned tree.
    """

    def __init__(
        self,
        solver: str,
        depth: int,
        time_limit: int,
        num_threads: None,
        verbose: bool,
    ) -> None:

        super().__init__(
            solver,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _tree_struc_variables(self) -> None:
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

    def _flow_variables(self) -> None:
        self._zeta = self._solver.add_vars(
            self._datapoints,
            self._tree.Nodes + self._tree.Leaves,
            self._treatments,
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

    def _define_variables(self) -> None:
        self._tree_struc_variables()
        self._flow_variables()

    def _tree_structure_constraints(self) -> None:
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

    def _flow_constraints(self) -> None:
        # z[i,n] = z[i,l(n)] + z[i,r(n)] + zeta[i,n]    forall i, n in Nodes
        for n in self._tree.Nodes:
            n_left = int(self._tree.get_left_children(n))
            n_right = int(self._tree.get_right_children(n))
            self._solver.add_constrs(
                (
                    self._z[i, n]
                    == self._z[i, n_left]
                    + self._z[i, n_right]
                    + self._solver.quicksum(
                        self._zeta[i, n, k] for k in self._treatments
                    )
                )
                for i in self._datapoints
            )

        for n in self._tree.Leaves:
            self._solver.add_constrs(
                self._solver.quicksum(self._zeta[i, n, k] for k in self._treatments)
                == self._z[i, n]
                for i in self._datapoints
            )

    def _arc_constraints(self) -> None:
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
            for k in self._treatments:
                self._solver.add_constrs(
                    self._zeta[i, n, k] <= self._w[n, k] for i in self._datapoints
                )

        self._solver.add_constrs(self._z[i, 1] == 1 for i in self._datapoints)

    def _define_constraints(self) -> None:
        self._tree_structure_constraints()
        self._flow_constraints()
        self._arc_constraints()
