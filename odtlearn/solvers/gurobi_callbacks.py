from gurobipy import GRB

from odtlearn.utils.callback_helpers import benders_callback, robust_benders_callback


class BendersCallback:
    """
    This class contains a function that is called by the Gurobi solver at
    every node through the branch-&-bound tree while we solve the model.

    We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every data-point we solve
    the sub-problem which is a minimum cut and check if g[i] <= value of
    sub-problem[i]. If this is violated we add the corresponding benders
    constraint as lazy constraint to the master problem and proceed.

    Parameters
    ----------
    store_search_progress_log : bool, optional (default=False)
        A flag to denote if the search progress log should be stored.

    Methods
    -------
    generate_constrs(model, depth=0, npass=0)
        Generate Benders' cuts at the current node in the branch-and-bound tree.
    """

    def __init__(self, store_search_progress_log=False):
        self.store_search_progress_log = store_search_progress_log

    def __call__(self, model, where) -> None:
        """
        Generate Benders' cuts at the current node in the branch-and-bound tree.

        Parameters
        ----------
        model : Model
            The optimization model.
        where : int
            Indicates where in the solve Gurobi is.
        """
        if self.store_search_progress_log:
            logging_callback(model, where)
        if where == GRB.Callback.MIPSOL:
            benders_callback(
                model, model._data["X"], model._data["obj"], model._data["solver"]
            )


class RobustBendersCallback:
    """
    A callback class for generating Benders' cuts in the Robust Tree optimization using Gurobi.

    This class contains a function that is called by the solver at every node in the
    branch-and-bound tree while solving the model. It checks for integer solutions to
    the master problem and generates Benders' cuts based on the subproblem solutions
    in the Robust Tree optimization.

    Parameters
    ----------
    store_search_progress_log : bool, optional (default=False)
        A flag to denote if the search progress log should be stored.

    Methods
    -------
    generate_constrs(model, depth=0, npass=0)
        Generate Benders' cuts at the current node in the branch-and-bound tree.
    """

    def __init__(self, store_search_progress_log=False):
        self.store_search_progress_log = store_search_progress_log

    def __call__(self, model, where):
        """
        Generate Benders' cuts at the current node in the branch-and-bound tree.

        Parameters
        ----------
        model : Model
            The optimization model.
        where : int
            Indicates where in the solve Gurobi is.
        """
        if self.store_search_progress_log:
            logging_callback(model, where)
        if where == GRB.Callback.MIPSOL:
            robust_benders_callback(
                model, model._data["X"], model._data["obj"], model._data["solver"]
            )


def logging_callback(model, where):
    """
    Helper function to store search progress log, in the same
    format as Python-MIP's search progress log. Restricted to
    log at least every 0.1 seconds as to not log too frequently.

    Parameters
    ----------
    model : Model
        The optimization model.
    """
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        if len(model._search_progress_log) > 0:
            last_time = model._search_progress_log[-1][0]
            if runtime - last_time < 0.1:
                # Too soon to log
                return

        best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
        best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)

        # Ensures logging of valid bounds
        if abs(best_obj) < 1e99 and abs(best_bound) < 1e99:
            model._search_progress_log.append((runtime, (best_obj, best_bound)))
