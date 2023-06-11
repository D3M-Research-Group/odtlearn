from gurobipy import LinExpr, Model
from gurobipy import quicksum as gb_quicksum

from odtlearn.utils.solver import Solver


class GurobiSolver(Solver):
    def __init__(self) -> None:
        self.model = Model()

    def get_var_value(self, objs, var_name=None):
        return self.model.getAttr("X", objs)

    def get_attr(self, name, objs):
        return self.model.getAttr(name, objs)

    def optimize(self, callback=False, callback_action=None):
        # self.model.update() # update is called when we call optimize
        if callback:
            if callback_action is not None:
                self.model.optimize(callback_action)
            else:
                raise TypeError("Must supply callback function if callback=True")
        else:
            self.model.optimize()

    def add_vars(
        self, *indices, lb=0.0, ub=float("inf"), obj=0.0, vtype="C", name: str = ""
    ):
        return self.model.addVars(
            *indices, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name
        )

    def add_constrs(self, cons_expr_tuple):
        self.model.addConstrs(cons_expr_tuple)

    def add_constr(self, cons_expr):
        self.model.addConstr(cons_expr)

    def lin_expr(self, arg1=0.0, arg2=None):
        return LinExpr(arg1, arg2)

    def set_objective(self, expr, sense):
        """
        Take the linear expression and set it as the objective for the problem
        """
        self.model.setObjective(expr, sense)

    def quicksum(self, terms):
        """
        Pass through function for Gurobi quicksum function
        """
        return gb_quicksum(terms)

    def store_data(self, key, value):
        try:
            getattr(self.model, "_data")
        except AttributeError:
            self.model._data = {}

        # if self.model._data is None:
        #     self.model._data = {}
        self.model._data[key] = value
