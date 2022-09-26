from gurobipy import LinExpr, Model

GUROBI = "gurobi"


class Solver:
    def __init__(self, model_name, solver="gurobi") -> None:
        self.solver = solver.lower()
        self.model = Model(model_name)

    def set_param(self, key, value):
        self.model.setParam(key, value)

    def addVars(
        self, *indices, lb=0.0, ub=float("inf"), obj=0.0, vtype="C", name: str = ""
    ):
        zipped_indices = self.zip_indices(indices)
        return self.model.addVars(
            zipped_indices, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name
        )

    def zip_indices(self, indices):
        to_zip = []
        # if given an integer, create range
        for elem in indices:
            if type(elem) is int:
                to_zip.append(range(elem))
            # if given float, coerce to integer
            if type(elem) is float:
                to_zip.append(range(int(elem)))
            # otherwise just pass the element to be zipped
            else:
                to_zip.append(elem)
        # TO-DO: check that all elements are the same length?
        return zip(to_zip)

    def addConstrs(self, cons_expr_tuple):
        for expr in cons_expr_tuple:
            self.addConstr(expr)

    def addConstr(self, cons_expr):
        self.model.addConstr(cons_expr)

    def obj_init(self, arg1=0.0):
        return LinExpr(arg1)

    def setObjective(self, expr, sense):
        self.model.setObjective(expr, sense)
