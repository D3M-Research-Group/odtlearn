from abc import ABC, abstractmethod

class Solver(ABC):
    """
    An abstract base class for interacting with the solver, including for setting up, optimizing, 
    and getting optimal values of a model.

    Parameters
    ----------
    solver : str
        The solver to use for the MIP formulation.

    Attributes
    ----------
    solver_name : str
        The name of the solver being used.

    Methods
    -------
    fit(X, y)
        Fit the optimal decision tree to the given training data.

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

    @abstractmethod
    def get_var_value(self, objs, var_name=None) -> dict:
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
        pass
    
    @abstractmethod
    def set_callback(self, callback_type):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def add_vars(
        self, *indices, lb=0.0, ub=float("inf"), obj=0.0, vtype="C", name: str = ""
    ) -> dict:
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
        pass

    @abstractmethod
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
        pass
    
    @abstractmethod
    def lin_expr(self, arg1=0.0, sense=None):
        """
        Initialize a linear expression object that the solver can read

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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def quicksum(self, terms):
        """
        Sums together a list of variables to make a linear expression

        Parameters
        ----------
        terms: List[mip.Variable]
            List of variables to be summed

        Returns
        -------
        LinExpr

        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_gap(self):
        pass

    @abstractmethod
    def get_objective_value(self):
        pass

    @abstractmethod
    def get_objective_bound(self):
        pass

    @abstractmethod
    def get_num_variables(self):
        pass

    @abstractmethod
    def get_num_int_variables(self):
        pass

    @abstractmethod
    def get_num_nonzeroes(self):
        pass

    @abstractmethod
    def get_num_soluions(self):
        pass

    @abstractmethod
    def get_num_constraints(self):
        pass

    @abstractmethod
    def set_time_limit(self, seconds):
        pass

    @abstractmethod
    def set_num_threads(self, num_threads):
        pass

    @abstractmethod
    def add_lazy_constraint(self, model, constr):
        pass

    @abstractmethod
    def get_callback_solution(self, model, var):
        pass

    @abstractmethod
    def get_search_progress_log(self):
        pass
