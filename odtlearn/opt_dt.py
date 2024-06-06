from abc import ABC, abstractmethod

from odtlearn.utils.solver import Solver
from odtlearn.utils.Tree import _Tree


class OptimalDecisionTree(ABC):
    def __init__(
        self, solver, depth=1, time_limit=60, num_threads=None, verbose=False
    ) -> None:
        """
        Parameters
        ----------
        depth : int, default=1
            A parameter specifying the depth of the tree
        time_limit : int, default=60
            The given time limit (in seconds) for solving the MIO problem
        num_threads: int, default=None
            The number of threads the solver should use. If no argument is supplied,
            Gurobi will use all available threads.
        verbose: bool, default=False
            A parameter passed to the solver to limit solver diagnostic output.
        """
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
        This function creates and return a Gurobi model based on the
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
    def optim_gap(self):
        return self._solver.model.gap

    @property
    def num_decision_vars(self):
        return self._solver.model.num_cols

    @property
    def num_integer_vars(self):
        return self._solver.model.num_int

    @property
    def num_non_zero(self):
        return self._solver.model.num_nz

    @property
    def num_solutions(self):
        return self._solver.model.num_solutions

    @property
    def num_constraints(self):
        return self._solver.model.num_rows

    @property
    def search_progress_log(self):
        return self._solver.model.search_progress_log
