from gurobipy import LinExpr, Model
from gurobipy import quicksum as gb_quicksum

from odtlearn.utils.solver import Solver


class GurobiSolver(Solver):
    """
    The Gurobi solver interface. This class contains functions for interacting
    with the solver for setting up, optimizing, and getting optimal values of a model.
    """

    def __init__(self) -> None:
        self.model = Model()

    def set_param(self, key, value):
        """
        A function for setting the Gurobi solver parameters.

        Parameters
        ----------
        key: str
            The name of the parameter to be set

        value: Any
            The value to set the Gurobi parameter to

        Returns
        -------
        None
        """
        self.model.setParam(key, value)

    def get_var_value(self, objs, var_name=None):
        """
        Get the value of a decision variable from a solved problem.

        Parameters
        ----------
        objs: dict
            A dictionary of the model variables

        var_name: str | None, default=None
            Ignored. Included for consistency of API.

        Returns
        -------
        A dict with the values of each variable from the solution
        """
        return self.model.getAttr("X", objs)

    def optimize(self, X, obj, solver, callback=False, callback_action=None, **kwargs):
        """Optimize the constructed model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Ignored. Included for consistency of API.

        obj : DecisionTree object
            Ignored. Included for consistency of API.

        solver : Solver object
            Ignored. Included for consistency of API.

        callback: bool, default=False
            Boolean specifying whether this model uses a callback when solving the problem

        callback_action: function object
            Function to be called when Gurobi reaches an integer solution

        kwargs: Additional arguments to be passed to the callback action

        Returns
        -------
        None
        """
        if callback:
            if callback_action is not None:
                self.model.optimize(callback_action)
            else:
                raise ValueError("Must supply callback action if callback=True")
        else:
            self.model.optimize()

    def add_vars(
        self, *indices, lb=0.0, ub=float("inf"), obj=0.0, vtype="C", name: str = ""
    ):
        """
        Create a dictionary with the decision variables with keys of the form
        {name}[(element of indices list)] and then add the variables to the model.

        Parameters
        ----------
        *indices: List
            Arbitrary list of indices to use to create the decision variables.

        lb: double, default=0.0
            Lower bound for new variable.

        ub: double, default=inf
            Upper bound for new variable.

        obj: double
            Objective coefficient for new variable.

        type: str, default="C"
            Variable type for new variable. Accepted values are "C", "B", "I"

        name: str, default=""
            Name used when creating dictionary storing variables.

        Returns
        -------
        Dictionary of new variable objects.
        """
        return self.model.addVars(
            *indices, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name
        )

    def add_constrs(self, cons_expr_tuple):
        """
        Add constraint expressions to the model.
        Thin wrapper around the Gurobi addConstrs function.

        Parameters
        ----------
        cons_expr_tuple: List[LinExpr]
            A list of constraint expressions to be added to the model.

        Returns
        -------
        None
        """
        self.model.addConstrs(cons_expr_tuple)

    def add_constr(self, cons_expr):
        """
        Add a constraint expression to the model.
        Thin wrapper around the Gurobi addConstr function.

        Parameters
        ----------
        cons_expr: LinExpr
            A constraint expression to be added to the model.

        Returns
        -------
        None
        """
        self.model.addConstr(cons_expr)

    def lin_expr(self, arg1=0.0, arg2=None):
        """
        Initialize a linear expression object
        Thin wrapper around the Gurobi LinExpr object.

        Parameters
        ----------
        arg1: double | Variable , default=0.0
            A constant or Variable to be used to initialize the linear expression

        arg2: Any | None, default=None
            Additional argument used to match Gurobi API.


        Returns
        -------
        Initalized LinExpr
        """
        return LinExpr(arg1, arg2)

    def set_objective(self, expr, sense):
        """
        Take the linear expression and set it as the objective for the problem.

        Parameters
        ----------

        expr: LinExpr
            The linear expression to be used as the objective for the problem.

        sense: str
            A string specifying whether the objective should be minimized (1 or GRB.MINIMIZE)
            or maximized (-1 or GRB.MAXIMIZE)

        Returns
        -------
        None
        """
        self.model.setObjective(expr, sense)

    def quicksum(self, terms):
        """
        Pass through function for Gurobi quicksum function

        Parameters
        ----------
        terms: List[gb.Variable]
            List of variables to be summed

        Returns
        -------
        LinExpr
        """
        return gb_quicksum(terms)

    def store_data(self, key, value):
        """
        Store data to be used in the callback action. For Gurobi, data can
        typically be stored as private attributes of the model (i.e., model._data_var).
        For consistency across solvers, we store the data in the model._data attribute
        as a dictionary.

        Parameters
        ----------
        key: str
            The name under which to store the data

        value: Any
            The values to be stored in the dictionary.

        Returns
        -------
        None
        """
        try:
            getattr(self.model, "_data")
        except AttributeError:
            self.model._data = {}

        # if self.model._data is None:
        #     self.model._data = {}
        self.model._data[key] = value
