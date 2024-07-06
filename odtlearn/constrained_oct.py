from abc import abstractmethod

from odtlearn.flow_oct_ms import FlowOCTMultipleSink


class ConstrainedOCT(FlowOCTMultipleSink):
    """
    A parent class for classification trees that incorporate
    side-constraints when fitting a tree (e.g., fairness constraints, etc.)

    Parameters
    ----------
    solver: str
        A string specifying the name of the solver to use
        to solve the MIP. Options are "Gurobi" and "CBC".
        If the CBC binaries are not found, Gurobi will be used by default.
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    depth : int, default= 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 60
        The given time limit (in seconds) for solving the MIO problem
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads
    """

    def __init__(
        self, solver, _lambda, depth, time_limit, num_threads, verbose
    ) -> None:

        super().__init__(solver, _lambda, depth, time_limit, num_threads, verbose)

    @abstractmethod
    def _define_side_constraints(self):
        pass

    def _define_constraints(self):
        super()._define_constraints()
        self._define_side_constraints()
