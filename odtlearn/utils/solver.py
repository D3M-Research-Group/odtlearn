from itertools import product

from mip import LinExpr, Model, xsum

from odtlearn.utils.mip_cbc import SolverCbc

GRB_CBC_CONST_MAP = {-1: "MAX", 1: "MIN"}


class Solver:
    """
    A wrapper class on top the python-mip Model and solver classes. This class contains functions for interacting
    with the solver for setting up, optimizing, and getting optimal values of a model.
    When using CBC, this class interacts with a slightly modified version of the SolverCbc class
    from the python-mip package.
    """

    def __init__(self, solver_name, verbose) -> None:
        self.solver_name = solver_name.lower()
        self.var_name_dict = {}

        if self.solver_name == "cbc":
            self.model = Model(solver_name="cbc")
            self.model.solver = SolverCbc(self.model, "cbc", self.model.sense, verbose)
        elif self.solver_name == "gurobi":
            self.model = Model(solver_name="gurobi")
        else:
            raise NotImplementedError(f"Solver {solver_name} not currently supported.")

    def get_var_value(self, objs, var_name=None):
        """
        Get the value of a decision variable from a solved problem.

        Parameters
        ----------
        objs: dict
            A dictionary of the model variables

        var_name: str | None, default=None
            The name supplied when the decision variable was initialized.

        Returns
        -------
        A dict with the values of each variable from the solution
        """
        result_dict = {}
        for key, _ in objs.items():
            name = self.var_name_dict[var_name][key]
            result_dict[key] = self.model.var_by_name(name).x
        return result_dict

    def optimize(self, X, obj, solver, callback=False, callback_action=None, **kwargs):
        """Optimize the constructed model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        obj : DecisionTree object
            A copy of the DecisionTree object that is passed to the callback action.

        solver : Solver object
            A copy of the Solver object that is passed to the callback action.

        callback: bool, default=False
            Boolean specifying whether this model uses a callback when solving the problem

        callback_action: mip.ConstrsGenerator object
            Function to be called when CBC reaches an integer solution

        kwargs: Additional arguments to be passed to the callback action

        Returns
        -------
        None
        """
        if callback:
            if callback_action is not None:
                self.model.lazy_constrs_generator = callback_action(
                    X, obj, solver, **kwargs
                )
                self.model.optimize()
            else:
                raise ValueError("Must supply callback action if callback=True")
        else:
            self.model.optimize()

    def prep_indices(self, *indices):
        """
        Helper function for prepping variable indices to generate
        decision variables with indices that mimic the structure of Gurobi-created
        decision variables

        Parameters
        ----------
        indices: List
            list of lists of indices.

        Returns
        -------
        A list with the generated indices.

        """
        prepped = []
        # if given an integer, create range
        for elem in indices:
            if type(elem) is int:
                prepped.append(list(range(elem)))
            # if given float, coerce to integer
            elif type(elem) is float:
                prepped.append(list(range(int(elem))))
            # otherwise just pass the element to be zipped
            else:
                prepped.append(elem)
        return prepped

    def add_vars(
        self, *indices, lb=0.0, ub=float("inf"), obj=0.0, vtype="C", name: str = ""
    ):
        """
        Create a dictionary with the decision variables with keys of the form
        {name}[(element of indices list)] and then add the variables to the model

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
        var_dict = {}
        name_element_dict = {}
        prepped = self.prep_indices(*indices)
        if len(prepped) > 1:
            for element in product(*prepped):
                name_element_dict[element] = (
                    f"{name}_{element}".replace("[", "_")
                    .replace("]", "_")
                    .replace(" ", "_")
                )
                var_dict[element] = self.model.add_var(
                    lb=lb,
                    ub=ub,
                    obj=obj,
                    var_type=vtype,
                    name=f"{name}_{element}".replace("[", "_")
                    .replace("]", "_")
                    .replace(" ", "_"),
                )
        else:
            for element in prepped[0]:
                name_element_dict[element] = (
                    f"{name}_{element}".replace("[", "_")
                    .replace("]", "_")
                    .replace(" ", "_")
                )
                var_dict[element] = self.model.add_var(
                    lb=lb,
                    ub=ub,
                    obj=obj,
                    var_type=vtype,
                    name=f"{name}_{element}".replace("[", "_")
                    .replace("]", "_")
                    .replace(" ", "_"),
                )
        self.var_name_dict[name] = name_element_dict
        return var_dict

    def add_constrs(self, cons_expr_tuple):
        """
        Add constraint expressions to the model.

        Parameters
        ----------
        cons_expr_tuple: List[LinExpr]
            A list of constraint expressions to be added to the model.

        Returns
        -------
        None
        """
        for cons in cons_expr_tuple:
            self.model.add_constr(cons)

    def add_constr(self, cons_expr):
        """
        Add a constraint expression to the model.

        Parameters
        ----------
        cons_expr: LinExpr
            A constraint expression to be added to the model.

        Returns
        -------
        None
        """
        self.model.add_constr(cons_expr)

    def lin_expr(self, arg1=0.0, sense=None):
        """
        Initialize a linear expression object

        Parameters
        ----------
        arg1: double | Variable , default=0.0
            A constant or Variable to be used to initialize the linear expression

        sense: str | None, default=None
            Argument for specifying whether the expression is to be minimized or maximized.


        Returns
        -------
        Initalized LinExpr
        """
        return LinExpr(const=arg1, sense=sense)

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
        self.model.objective = expr
        if type(sense) is int:
            mapped_sense = GRB_CBC_CONST_MAP.get(sense, None)
            if mapped_sense is None:
                raise ValueError(f"Invalid objective type: {sense}.")
            else:
                sense = mapped_sense
        elif type(sense) is str:
            if sense not in ["MAX", "MIN"]:
                raise ValueError(f"Invalid objective type: {sense}.")
        else:
            raise TypeError("Objective sense must be integer or string.")

        self.model.sense = sense

    def quicksum(self, terms):
        """
        Pass through function for python-mip quicksum function

        Parameters
        ----------
        terms: List[mip.Variable]
            List of variables to be summed

        Returns
        -------
        LinExpr

        """
        return xsum(terms)

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
        self.model._data[key] = value
