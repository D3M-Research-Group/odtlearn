from abc import ABC, abstractmethod
from collections.abc import Mapping, Iterable
from typing import Any, Optional, Union


class Solver(ABC):
    """
    An abstract base class for interacting with the solver, including for setting up, optimizing,
    and getting optimal values of a model.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print verbose output from the solver.

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly.
    Subclasses must implement the following abstract methods:

    * `add_vars()`
    * `add_constr()`
    * `set_objective()`
    * `set_time_limit()`
    * `set_num_threads()`
    * `set_callback()`
    * `store_data()`
    * `optimize()`
    * `add_lazy_constraint()`
    * `get_callback_solution()`
    * `get_var_value()`
    * `get_gap()`
    * `get_objective_value()`
    * `get_objective_bound()`
    * `get_num_constraints()`
    * `get_search_progress_log()`
    * `lin_expr()`
    * `quicksum()`


    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    # Methods to setup optimization problem
    @abstractmethod
    def add_vars(
        self,
        *indices,
        lb: float = 0.0,
        ub: float = float("inf"),
        obj: float = 0.0,
        vtype: str = "C",
        name: str = "",
    ) -> Mapping:
        """
        Create a dictionary with the decision variables with keys of the form
        {name}[(element of indices list)] and then add the variables to the model

        Parameters
        ----------
        *indices: Iterable
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
        cons_expr: Any
            A constraint expression to be added to the model. Type is dependent on
            solver.

        Returns
        -------
        None
        """
        pass

    def add_constrs(self, cons_expr_tuple: Iterable):
        """
        Add constraint expressions to the model.

        Parameters
        ----------
        cons_expr_tuple: Iterable
            A list of constraint expressions to be added to the model.

        Returns
        -------
        None
        """
        for cons in cons_expr_tuple:
            self.add_constr(cons)

    @abstractmethod
    def set_objective(self, expr, sense: Union[str, int]):
        """
        Take the linear expression and set it as the objective for the problem.

        Parameters
        ----------

        expr: Any
            The linear expression to be used as the objective for the problem.

        sense: str or int
            A string specifying whether the objective should be minimized (1 or GRB.MINIMIZE)
            or maximized (-1 or GRB.MAXIMIZE)

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def set_time_limit(self, seconds: int):
        """
        Set the time limit for the solver.

        Parameters
        ----------

        seconds : int
            Time limit to be set, in seconds

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def set_num_threads(self, num_threads: int):
        """
        Set the number of threads used by the solver.

        Parameters
        ----------

        num_threads : int
            Number of threads

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def set_callback(self, callback_type: str):
        """
        Set a callback for the solver.

        Parameters
        ----------

        callback_type : str
            Name of callback to add to the solver.
            Possible values are currently "benders" and "robust_benders"

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def store_data(self, key: str, value):
        """
        Store data to be used in the callback action.
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

    # Methods used to initialte and use during solve
    @abstractmethod
    def optimize(self):
        """Optimize the constructed model

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def add_lazy_constraint(self, model, constr):
        """Add a lazy constraint to the model

        Parameters
        ----------
        model: Any
            The model object to add the lazy constraint to.

        constr: Any
            The lazy constraint to add to the model.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_callback_solution(self, model, var) -> Any:
        """Obtain the value of a variable in a solution within a callback

        Parameters
        ----------
        model: Any
            The model object containing the callback solution.

        var: Any
            The variable object to get the value of.

        Returns
        -------
        The callback solution.
        """
        pass

    # Methods to get solutions and solver statistics
    @abstractmethod
    def get_var_value(self, objs: Mapping, var_name: str = None) -> Mapping:
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
    def get_gap(self) -> Optional[float]:
        """
        Get the MIP gap of the solve after optimization.

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def get_objective_value(self) -> Optional[float]:
        """
        Get the objective value of the solution.

        Returns
        -------
        float or None
        """
        pass

    @abstractmethod
    def get_objective_bound(self) -> Optional[float]:
        """
        Get the best known objective bound of the solution. This is a
        lower bound for minimization and upper bound for maximization.

        Returns
        -------
        float or None
        """
        pass

    @abstractmethod
    def get_num_variables(self) -> int:
        """
        Get the number of variables in the optimization model.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def get_num_int_variables(self) -> int:
        """
        Get the number of integer-valued variables in the solution.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def get_num_nonzeroes(self) -> int:
        """
        Get the number of nonzeros in the constraint matrix of the model.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def get_num_soluions(self) -> int:
        """
        Get the number of solutions found during the solve.

        Returns
        -------
        int
        """

    @abstractmethod
    def get_num_constraints(self) -> int:
        """
        Get the number of constraints in the model.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def get_search_progress_log(self) -> Iterable[tuple]:
        """
        Get the number of solutions found during the solve.

        Returns
        -------
        List of tuples of the form (time, (lower_bound, upper_bound))
        """
        pass

    # Methods for variable and constraint construction
    @abstractmethod
    def lin_expr(self, arg1: float = 0.0) -> Any:
        """
        Initialize a linear expression object that the solver can read

        Parameters
        ----------
        arg1: double | Variable , default=0.0
            A constant or Variable to be used to initialize the linear expression

        Returns
        -------
        Any, solver-dependent
        """
        pass

    @abstractmethod
    def quicksum(self, terms: Iterable) -> Any:
        """
        Sums together a list of variables to make a linear expression

        Parameters
        ----------
        terms: list
            list of variables to be summed

        Returns
        -------
        Any, solver-dependent

        """
        pass
