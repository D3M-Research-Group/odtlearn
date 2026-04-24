from mip import ConstrsGenerator, Model

from odtlearn.utils.callback_helpers import (
    benders_callback,
    robust_benders_callback
)

class BendersCallback(ConstrsGenerator):
    """
    This class contains a function that is called by the solver at
    every node through the branch-&-bound tree while we solve the model.

    We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every data-point we solve
    the sub-problem which is a minimum cut and check if g[i] <= value of
    sub-problem[i]. If this is violated we add the corresponding benders
    constraint as lazy constraint to the master problem and proceed.
    Whenever we have no violated constraint, it means that we have found
    the optimal solution.

    """

    def __init__(self, data):
        self.data = data

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0) -> None:
        """
        Generate Benders' cuts at the current node in the branch-and-bound tree.

        Parameters
        ----------
        model : Model
            The optimization model.
        depth : int, optional
            The depth of the current node in the branch-and-bound tree.
        npass : int, optional
            The pass number in the branch-and-bound process.
        """
        benders_callback(model, self.data['X'], self.data['obj'], self.data['solver'])


class RobustBendersCallback(ConstrsGenerator):
    """
    A callback class for generating Benders' cuts in the Robust Tree optimization.

    This class contains a function that is called by the solver at every node in the
    branch-and-bound tree while solving the model. It checks for integer solutions to
    the master problem and generates Benders' cuts based on the subproblem solutions
    in the Robust Tree optimization.

    Parameters
    ----------
    X : DataFrame
        The input data.
    obj : object
        The main model object.
    solver : Solver
        The solver object used for solving the optimization problem.
    b : dict
        The dictionary of decision variables representing the branching decisions.
    w : dict
        The dictionary of decision variables representing the class assignments.
    t : dict
        The dictionary of decision variables representing the subproblem objective values.

    Methods
    -------
    generate_constrs(model, depth=0, npass=0)
        Generate Benders' cuts at the current node in the branch-and-bound tree.
    """

    def __init__(self, data):
        self.data = data

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0):
        """
        Generate Benders' cuts at the current node in the branch-and-bound tree.

        Parameters
        ----------
        model : Model
            The optimization model.
        depth : int, optional
            The depth of the current node in the branch-and-bound tree.
        npass : int, optional
            The pass number in the branch-and-bound process.
        """
        robust_benders_callback(model, self.data['X'], self.data['obj'], self.data['solver'])
