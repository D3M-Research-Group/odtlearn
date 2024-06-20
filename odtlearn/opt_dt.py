from abc import ABC, abstractmethod

import mip

from odtlearn.utils.solver import Solver
from odtlearn.utils.Tree import _Tree


class OptimalDecisionTree(ABC):
    """
    An abstract base class for learning optimal decision trees using mixed-integer programming.

    Parameters
    ----------
    solver : str
        The solver to use for the MIP formulation. Currently, only "gurobi" and "CBC" are supported.
    depth : int, default=1
        The maximum depth of the tree to be learned.
    time_limit : int, default=60
        The time limit (in seconds) for solving the MIP formulation.
    num_threads : int, optional
        The number of threads the solver should use. If not specified,
        solver uses all available threads
    verbose : bool, default=False
        Whether to print verbose output from the solver.

    Attributes
    ----------
    solver_name : str
        The name of the solver being used.
    optim_gap : float
        The optimality gap considering the cost of the best solution found.
    objective_value : float
        The objective function value of the solution found or None if the model was not optimized.
    objective_bound : float
        A valid estimate computed for the optimal solution cost. The bound is equal to the objective
        value if the optimal solution is found.
    num_decision_vars : int
        The number of decision variables in the model.
    num_integer_vars : int
        The number of integer variables in the model.
    num_non_zero : int
        The number of non-zeros in the constraint matrix.
    num_solutions : int
        The number of solutions found during the optimization.
    num_constraints : int
        The number of constraints in the model.
    search_progress_log : mip.ProgressLog
        The log of bound improvements during the optimization.
    store_search_progress_log : bool
        Whether the search progress log should be stored or not.

    Methods
    -------
    fit(X, y)
        Fit the optimal decision tree to the given training data.
    predict(X)
        Make predictions using the fitted optimal decision tree.

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly.
    Subclasses must implement the following abstract methods:

    * `_define_variables()`
    * `_define_constraints()`
    * `_define_objective()`
    * `fit()`
    * `predict()`
    """

    def __init__(
        self, solver, depth=1, time_limit=60, num_threads=None, verbose=False
    ) -> None:

        self.solver_name = solver
        self._depth = depth
        self._time_limit = time_limit
        self._num_threads = num_threads
        self._verbose = verbose

        self._tree = _Tree(self._depth)
        self._time_limit = time_limit

        self._solver = Solver(self.solver_name, verbose)

        if num_threads is not None:
            self._solver.model.threads = num_threads
        self._solver.model.max_seconds = time_limit

    def __repr__(self):
        rep = (
            f"{type(self).__name__}(solver={self.solver_name},"
            f"depth={self._depth},"
            f"time_limit={self._time_limit},"
            f"num_threads={self._num_threads},"
            f"verbose={self._verbose})"
        )
        return rep

    @abstractmethod
    def _define_variables(self):
        pass

    @abstractmethod
    def _define_constraints(self):
        pass

    @abstractmethod
    def _define_objective(self):
        pass

    def _create_main_problem(self):
        """
        This function creates and return a model based on the
        variables, constraints, and objective defined within a subclass
        """
        self._define_variables()
        self._define_constraints()
        self._define_objective()

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @property
    def optim_gap(self) -> float:
        """
        The optimality gap considering the cost of the best solution found.
        """
        return self._solver.model.gap

    @property
    def objective_value(self) -> float:
        """
        Objective function value of the solution found or None if the model was not optimized.
        """

        return self._solver.model.objective_value

    @property
    def objective_bound(self) -> float:
        """
        A valid estimate computed for the optimal solution cost.
        The bound is equal to the objective value if the optimal solution is found.
        """
        return self._solver.model.objective_bound

    @property
    def num_decision_vars(self) -> int:
        """Number of decision variables in the model."""
        return self._solver.model.num_cols

    @property
    def num_integer_vars(self) -> int:
        """Number of integer variables in the model."""
        return self._solver.model.num_int

    @property
    def num_non_zero(self) -> int:
        """Number of non-zeros in the constraint matrix."""
        return self._solver.model.num_nz

    @property
    def num_solutions(self) -> int:
        """Number of solutions found during the optimization."""
        return self._solver.model.num_solutions

    @property
    def num_constraints(self) -> int:
        """Number of constraints in the model."""
        return self._solver.model.num_rows

    @property
    def search_progress_log(self) -> mip.ProgressLog:
        """Log of bound improvements during the optimization."""
        return self._solver.model.search_progress_log

    @property
    def store_search_progress_log(self) -> bool:
        """
        Boolean for whether the search progress log should be stored or not.
        As in python-mip this is disabled by default.
        Enable storing the search progress log to analyze bound improvements over time.
        """
        return self._solver.model.__store_search_progress_log

    @store_search_progress_log.setter
    def store_search_progress_log(self, store: bool):
        self._solver.model.store_search_progress_log = store
