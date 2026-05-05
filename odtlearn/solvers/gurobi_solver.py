from gurobipy import Model, GRB, quicksum, LinExpr, Env

from odtlearn.solvers.solver import Solver

from odtlearn.callbacks.gurobi_callbacks import BendersCallback, RobustBendersCallback, logging_callback

class GurobiSolver(Solver):
    """
    A wrapper class on top of gurobipy Model.
    """

    def __init__(self, verbose) -> None:
        self.solver_name = "gurobi"

        env = Env(empty=True)
        if not verbose:
            env.setParam("OutputFlag", 0)
        env.start()

        self.model = Model()
        self.callback = None
        self.store_search_progress_log = False
        self.model._search_progress_log = []
        

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
        return {key: var.X for key, var in objs.items()}

    def set_callback(self, callback_type):
        """
        Set the callback used when solving.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        obj : DecisionTree object
            A copy of the DecisionTree object that is passed to the callback action.

        callback: bool, default=False
            Boolean specifying whether this model uses a callback when solving the problem

        callback_type: string
            Indicates which callback to use

        kwargs: Additional arguments to be passed to the callback action

        Returns
        -------
        None
        """
        if callback_type == "benders":
            self.callback = BendersCallback()
        elif callback_type == "robust_benders":
            self.callback = RobustBendersCallback()
        else:
            raise ValueError("callback_type not supported")
        self.model.setParam('LazyConstraints', 1)

    def optimize(self):
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
        if self.callback is not None:
            self.callback.store_search_progress_log = self.store_search_progress_log
            self.model.optimize(self.callback)
        elif self.store_search_progress_log:
            self.model.optimize(logging_callback)
        else:
            self.model.optimize()

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
        var_dict = self.model.addVars(*indices, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name)
        # print("🤢")
        # print(var_dict)
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
            self.model.addConstr(cons)

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
        self.model.addConstr(cons_expr)

    def lin_expr(self, arg1=0.0):
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
        return LinExpr(arg1)

    def set_objective(self, expr, sense):
        """
        Take the linear expression and set it as the objective for the problem.

        Parameters
        ----------

        expr: LinExpr
            The linear expression to be used as the objective for the problem.

        sense: str
            A string specifying whether the objective should be minimized (1)
            or maximized (-1)

        Returns
        -------
        None
        """
        if type(sense) is int:
            if sense not in [GRB.MAXIMIZE, GRB.MINIMIZE]:
                raise ValueError(f"Invalid objective type: {sense}.")
        elif type(sense) is str:
            if sense == "MAX":
                sense = GRB.MAXIMIZE
            elif sense == "MIN":
                sense = GRB.MINIMIZE
            else:
                raise ValueError(f"Invalid objective type: {sense}.")
        else:
            raise TypeError("Objective sense must be integer or string.")
        self.model.setObjective(expr, sense)

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
        return quicksum(terms)

    def store_data(self, key, value):
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
        try:
            getattr(self.model, "_data")
        except AttributeError:
            self.model._data = {}
        self.model._data[key] = value

    def get_gap(self):
        return self.model.MIPGap

    def get_objective_value(self):
        return self.model.ObjVal

    def get_objective_bound(self):
        return self.model.ObjBound

    def get_num_variables(self):
        return self.model.NumVars

    def get_num_int_variables(self):
        return self.model.NumIntVars

    def get_num_nonzeroes(self):
        return self.model.NumNZs

    def get_num_soluions(self):
        return self.model.SolCount

    def get_num_constraints(self):
        return self.model.NumConstrs

    def set_time_limit(self, seconds):
        self.model.setParam('TimeLimit', seconds)

    def set_num_threads(self, num_threads):
        self.model.setParam('Threads', num_threads)
    
    def add_lazy_constraint(self, model, constr):
        model.cbLazy(constr)
    
    def get_callback_solution(self, model, var):
        return model.cbGetSolution(var)
    
    def get_search_progress_log(self):
        return self.model._search_progress_log
    
