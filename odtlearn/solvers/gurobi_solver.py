from gurobipy import Model, GRB, quicksum, LinExpr, Env, tupledict, Var

from odtlearn.solvers.solver import Solver

from odtlearn.solvers.gurobi_callbacks import (
    BendersCallback,
    RobustBendersCallback,
    logging_callback,
)


class GurobiSolver(Solver):
    """
    A wrapper class on top of the gurobipy Model. This class contains functions for interacting
    with the solver for setting up, optimizing, and getting optimal values of a model.
    """

    def __init__(self, verbose):
        env = Env(empty=True)
        if not verbose:
            env.setParam("OutputFlag", 0)
        env.start()

        self.model = Model()
        self.callback = None
        self.store_search_progress_log = False
        self.model._search_progress_log = []

    def get_var_value(self, objs, var_name=None):
        return {key: var.X for key, var in objs.items()}

    def set_callback(self, callback_type):
        if callback_type == "benders":
            self.callback = BendersCallback()
        elif callback_type == "robust_benders":
            self.callback = RobustBendersCallback()
        else:
            raise ValueError("callback_type not supported")
        self.model.setParam("LazyConstraints", 1)

    def optimize(self):
        if self.callback is not None:
            self.callback.store_search_progress_log = self.store_search_progress_log
            self.model.optimize(self.callback)
        elif self.store_search_progress_log:
            self.model.optimize(logging_callback)
        else:
            self.model.optimize()

    def add_vars(
        self, *indices, lb=0.0, ub=float("inf"), obj=0.0, vtype="C", name: str = ""
    ) -> tupledict:
        var_dict = self.model.addVars(
            *indices, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name
        )
        return var_dict

    def add_constr(self, cons_expr: LinExpr):
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

    def lin_expr(self, arg1=0.0) -> LinExpr:
        return LinExpr(arg1)

    def set_objective(self, expr: LinExpr, sense):
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

    def quicksum(self, terms: list) -> LinExpr:
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
        if self.model.SolCount > 0:
            return self.model.ObjVal
        return None

    def get_objective_bound(self):
        if self.model.SolCount > 0:
            return self.model.ObjBound
        return None

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
        self.model.setParam("TimeLimit", seconds)

    def set_num_threads(self, num_threads):
        self.model.setParam("Threads", num_threads)

    def add_lazy_constraint(self, model: Model, constr: LinExpr):
        model.cbLazy(constr)

    def get_callback_solution(self, model: Model, var: Var):
        return model.cbGetSolution(var)

    def get_search_progress_log(self) -> list:
        return self.model._search_progress_log
