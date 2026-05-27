from mip import ConstrsGenerator, Model

from odtlearn.utils.callback_helpers import (
    benders_callback,
    robust_benders_callback
)

class BendersCallback(ConstrsGenerator):
    """
    This class contains a function that is called by the CBC solver at
    every node through the branch-&-bound tree while we solve the model.

    We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every data-point we solve
    the sub-problem which is a minimum cut and check if g[i] <= value of
    sub-problem[i]. If this is violated we add the corresponding benders
    constraint as lazy constraint to the master problem and proceed.

    Parameters
    ----------
    data : dict
        A dictionary containing the data needed in the callback (i.e., the
        training data, the BendersOCT object, and the CBCSolver object)

    Methods
    -------
    generate_constrs(model, depth=0, npass=0)
        Generate Benders' cuts at the current node in the branch-and-bound tree.
    """

    def __init__(self, data):
        self.data = data

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0) -> None:
        """
        Generate Benders' cuts at the current node in the branch-and-bound tree.

        Parameters
        ----------
        model : mip.Model
            The optimization model
        depth : int, optional
            The depth of the current node in the branch-and-bound tree.
        npass : int, optional
            The pass number in the branch-and-bound process.
        """
        benders_callback(model, self.data['X'], self.data['obj'], self.data['solver'])


class RobustBendersCallback(ConstrsGenerator):
    """
    A callback class for generating Benders' cuts in the Robust Tree optimization using CBC.

    This class contains a function that is called by the solver at every node in the
    branch-and-bound tree while solving the model. It checks for integer solutions to
    the master problem and generates Benders' cuts based on the subproblem solutions
    in the Robust Tree optimization.

    Parameters
    ----------
    data : dict
        A dictionary containing the data needed in the callback (i.e., the
        training data, the RobustOCT object, and the CBCSolver object)

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
