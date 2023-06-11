from itertools import product

from mip import ConstrsGenerator, LinExpr, Model, xsum

from odtlearn.utils.callbacks import benders_subproblem, get_cut_integer
from odtlearn.utils.mip_cbc import SolverCbc
from odtlearn.utils.solver import Solver

GRB_CBC_CONST_MAP = {-1: "MAX", 1: "MIN"}


class CBCSolver(Solver):
    def __init__(self) -> None:
        self.model = Model(solver_name="cbc")
        self.model.solver = SolverCbc(self.model, "cbc", self.model.sense)
        self.var_name_dict = {}

    def get_var_value(self, objs, var_name=None):
        result_dict = {}
        for key, _ in objs.items():
            name = self.var_name_dict[var_name][key]
            result_dict[key] = self.model.var_by_name(name).x
        return result_dict

    def get_attr(self, name, objs, var_name=None):
        """
        Mimic the getAttr functionality from Gurobi to get the value
        of a variable in the current solution. Raises an implementation
        error if an attribute name besides "X" is given.
        """
        if name == "X":
            result_dict = {}
            for key, value in objs.items():
                result_dict[key] = self.model.var_by_name(value)
            return result_dict
        else:
            raise NotImplementedError(f"Unable to get attribute {name} for CBC solver")

    def optimize(self, *vars, X, obj, solver, callback=False):
        if callback:
            self.model.lazy_constrs_generator = BendersCallback(*vars, X, obj, solver)
            self.model.optimize()
        else:
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
        name_element_dict = {}
        prepped = self.prep_indices(*indices)
        if len(prepped) > 1:
            for element in product(*prepped):
                name_element_dict[element] = f"{name}[{element}]"
                var_dict[element] = self.model.add_var(
                    lb=lb, ub=ub, obj=obj, var_type=vtype, name=f"{name}[{element}]"
                )
        else:
            for element in prepped[0]:
                name_element_dict[element] = f"{name}[{element}]"
                var_dict[element] = self.model.add_var(
                    lb=lb, ub=ub, obj=obj, var_type=vtype, name=f"{name}[{element}]"
                )
        self.var_name_dict[name] = name_element_dict
        return var_dict

    def add_constrs(self, cons_expr_tuple):
        for cons in cons_expr_tuple:
            self.model.add_constr(cons)

    def add_constr(self, cons_expr):
        self.model.add_constr(cons_expr)

    def lin_expr(self, arg1=0.0, sense=None):
        return LinExprWrapper(const=arg1, sense=sense)

    def set_objective(self, expr, sense):
        """
        Take the linear expression and set it as the objective for the problem
        """
        self.model.objective = expr
        if type(sense) is int:
            mapped_sense = GRB_CBC_CONST_MAP.get(sense, None)
            if mapped_sense is None:
                raise ValueError(f"Invalid objective type: {sense}")
            else:
                sense = mapped_sense
        self.model.sense = sense

    def quicksum(self, terms):
        """
        Pass through function for Gurobi quicksum function
        """
        return xsum(terms)

    def store_data(self, key, value):
        try:
            getattr(self.model, "_data")
        except AttributeError:
            self.model._data = {}

        # if self.model._data is None:
        #     self.model._data = {}
        self.model._data[key] = value


class LinExprWrapper(LinExpr):
    def __init__(self, const, sense):
        super().__init__(const=const, sense=sense)

    def add(self, other):
        self.__add__(other)


class BendersCallback(ConstrsGenerator):
    def __init__(self, g, b, p, w, X, obj, solver):
        self.X = X
        self.obj = obj
        self.solver = solver
        self.g = g
        self.p = p
        self.b = b
        self.w = w

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0):
        g_trans, p_trans, b_trans, w_trans = (
            {k: model.translate(v).x for k, v in self.g.items()},
            {k: model.translate(v).x for k, v in self.p.items()},
            {k: model.translate(v).x for k, v in self.b.items()},
            {k: model.translate(v).x for k, v in self.w.items()},
        )
        print(f"g: {g_trans}")
        print(f"b: {b_trans}")
        print(f"p: {p_trans}")
        print(f"w: {w_trans}")
        for i in self.X.index:
            g_threshold = 0.5
            if g_trans[i] > g_threshold:
                subproblem_value, left, right, target = benders_subproblem(
                    self.obj, b_trans, p_trans, w_trans, i
                )
                print(subproblem_value, left, right, target)
                if subproblem_value == 0:
                    lhs = get_cut_integer(
                        self.solver,
                        self.obj,
                        left,
                        right,
                        target,
                        i,
                    )
                    print(lhs)
                    new_constr = lhs
                    print(new_constr)
                    new_constr.sense = "<"
                    print(new_constr.sense)
                    print(new_constr.const)
                    print(new_constr.x)

                    # lhs.violation = 0
                    # model.add_constr(new_constr)
                    model += new_constr


# def get_left_exp_integer(solver, b, X, X_labels, n, i):
#     lhs = solver.quicksum(-1 * b[n, f] for f in X_labels if X.at[i, f] == 0)

#     return lhs


# def get_right_exp_integer(solver, b, X, X_labels, n, i):
#     lhs = solver.quicksum(-1 * b[n, f] for f in X_labels if X.at[i, f] == 1)

#     return lhs


# def get_target_exp_integer(main_grb_obj, w, n, i):
#     label_i = main_grb_obj._y[i]
#     lhs = -1 * w[n, label_i]
#     return lhs


# def get_cut_integer(solver, g, b, w, X, X_labels, main_grb_obj, left, right, target, i):
#     lhs = solver.lin_expr(0.0, sense="<")
#     lhs += g[i]
#     for n in left:
#         tmp_lhs = get_left_exp_integer(solver, b, X, X_labels, n, i)
#         # lhs = lhs + tmp_lhs
#         lhs += tmp_lhs

#     for n in right:
#         tmp_lhs = get_right_exp_integer(solver, b, X, X_labels, n, i)
#         # lhs = lhs + tmp_lhs
#         lhs += tmp_lhs

#     for n in target:
#         tmp_lhs = get_target_exp_integer(main_grb_obj, w, n, i)
#         # lhs = lhs + tmp_lhs
#         lhs += tmp_lhs

#     return lhs
