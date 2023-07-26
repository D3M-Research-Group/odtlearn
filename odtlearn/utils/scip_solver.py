import itertools

from pyscipopt import Expr, Model
from pyscipopt import quicksum as scip_quicksum

from odtlearn.utils.solver import Solver

GRB_CONST_MAP = {-1: "maximize", 1: "minimize"}


class SCIPSolver(Solver):
    def __init__(self):
        self.model = Model()

    def get_attr(self, name, objs):
        """
        Mimic the getAttr functionality from Gurobi to get the value
        of a variable in the current solution. Raises an implementation
        error if an attribute name besides "X" is given.
        """
        if name == "X":
            result_dict = {}
            for key, value in objs.items():
                result_dict[key] = self.model.getVal(value)
            return result_dict
        else:
            raise NotImplementedError(f"Unable to get attribute {name} for SCIP solver")

    def optimize(self):
        self.model.optimize()

    def prep_indices(self, *indices):
        """
        indices: list of lists
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
        # TO-DO: check that all elements are the same length?
        return prepped

    def add_vars(
        self, *indices, lb=0.0, ub=float("inf"), obj=0.0, vtype="C", name: str = ""
    ):
        var_dict = {}
        prepped = self.prep_indices(*indices)
        if len(prepped) > 1:
            for element in itertools.product(*prepped):
                var_dict[element] = self.model.addVar(
                    lb=lb, ub=ub, obj=obj, vtype=vtype, name=f"{name}[{element}]"
                )
        else:
            for element in prepped[0]:
                var_dict[element] = self.model.addVar(
                    lb=lb, ub=ub, obj=obj, vtype=vtype, name=f"{name}[{element}]"
                )
        return var_dict

    def add_constr(self, cons_expr):
        self.model.addCons(cons_expr)

    def add_constrs(self, cons_expr_tuple):
        self.model.addConss(cons_expr_tuple)

    def lin_expr(self, arg1=0.0):
        """
        arg1 is ignored for now.
        """
        return ExprWrapper()

    def set_objective(self, expr, sense):
        if type(sense) is int:
            mapped_sense = GRB_CONST_MAP.get(sense, None)
            if mapped_sense is None:
                raise ValueError(f"Invalid objective type: {sense}")
            else:
                sense = mapped_sense
        self.model.setObjective(expr, sense)

    def quicksum(self, terms):
        return scip_quicksum(terms)


class ExprWrapper(Expr):
    def __init__(self):
        super().__init__()

    def add(self, other):
        self += other
