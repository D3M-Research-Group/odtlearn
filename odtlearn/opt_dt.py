from abc import ABC, abstractmethod

from odtlearn.utils.cbc_solver import CBCSolver
from odtlearn.utils.gb_solver import GurobiSolver
from odtlearn.utils.scip_solver import SCIPSolver
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
        """
        self.solver_name = solver
        self._depth = depth
        self._time_limit = time_limit
        self._num_threads = num_threads
        self._verbose = verbose

        self._tree = _Tree(self._depth)
        self._time_limit = time_limit

        if solver.lower() == "gurobi":
            self._solver = GurobiSolver()
        elif solver.lower() == "scip":
            self._solver = SCIPSolver()
        elif solver.lower() == "cbc":
            self._solver = CBCSolver()
        else:
            raise NotImplementedError(
                f"Interface for {solver} not currently supported."
            )

        if not verbose:
            # supress all logging
            if self._solver.__class__.__name__ == "GurobiSolver":
                self._solver.set_param("OutputFlag", 0)
            elif self._solver.__class__.__name__ == "CBCSolver":
                self._solver.model.verbose = 0
            else:
                self._solver.set_param("display/verblevel", 0)
        if num_threads is not None:
            if self._solver.__class__.__name__ == "GurobiSolver":
                self._solver.set_param("Threads", num_threads)
            elif self._solver.__class__.__name__ == "CBCSolver":
                self._solver.model.threads = num_threads
            else:
                self._solver.set_param("lp/threads", num_threads)
        if self._solver.__class__.__name__ == "GurobiSolver":
            self._solver.set_param("TimeLimit", time_limit)
        elif self._solver.__class__.__name__ == "CBCSolver":
            self._solver.model.max_seconds = time_limit
        else:
            self._solver.set_param("limits/time", time_limit)

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
