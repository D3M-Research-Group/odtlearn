from itertools import product

from mip import LinExpr, Model, xsum

from odtlearn.utils.mip_cbc import SolverCbc

from odtlearn.solvers.solver import Solver
from odtlearn.solvers.cbc_callbacks import BendersCallback, RobustBendersCallback

GRB_CBC_CONST_MAP = {-1: "MAX", 1: "MIN"}


class CBCSolver(Solver):
    """
    A wrapper class on top the python-mip Model and solver classes. This class contains functions for interacting
    with the solver for setting up, optimizing, and getting optimal values of a model.
    When using CBC, this class interacts with a slightly modified version of the SolverCbc class
    from the python-mip package.
    """

    def __init__(self, verbose):
        self.var_name_dict = {}
        self.callback = None
        self.model = Model(solver_name="cbc")
        self.model.solver = SolverCbc(self.model, "cbc", self.model.sense, verbose)
        self.store_search_progress_log = False

    def get_var_value(self, objs, var_name=None):
        result_dict = {}
        for key, _ in objs.items():
            name = self.var_name_dict[var_name][key]
            result_dict[key] = self.model.var_by_name(name).x
        return result_dict

    def set_callback(self, callback_type):
        try:
            getattr(self.model, "_data")
        except AttributeError:
            self.model._data = {}
        if callback_type == "benders":
            self.callback = BendersCallback(self.model._data)
        elif callback_type == "robust_benders":
            self.callback = RobustBendersCallback(self.model._data)
        else:
            raise ValueError("callback_type not supported")

    def optimize(self):
        if self.store_search_progress_log:
            self.model.store_search_progress_log = True
        if self.callback is not None:
            self.model.lazy_constrs_generator = self.callback
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

    def add_constr(self, cons_expr: LinExpr):
        self.model.add_constr(cons_expr)

    def lin_expr(self, arg1=0.0, sense: str = "") -> LinExpr:
        return LinExpr(const=arg1, sense=sense)

    def set_objective(self, expr: LinExpr, sense):
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
        return xsum(terms)

    def store_data(self, key, value):
        try:
            getattr(self.model, "_data")
        except AttributeError:
            self.model._data = {}
        self.model._data[key] = value

    def get_gap(self):
        return self.model.gap

    def get_objective_value(self):
        return self.model.objective_value

    def get_objective_bound(self):
        return self.model.objective_bound

    def get_num_variables(self):
        return self.model.num_cols

    def get_num_int_variables(self):
        return self.model.num_int

    def get_num_nonzeroes(self):
        return self.model.num_nz

    def get_num_soluions(self):
        return self.model.num_solutions

    def get_num_constraints(self):
        return self.model.num_rows

    def set_time_limit(self, seconds):
        self.model.max_seconds = seconds

    def set_num_threads(self, num_threads):
        self.model.threads = num_threads

    def add_lazy_constraint(self, model: Model, constr: LinExpr):
        model += constr

    def get_callback_solution(self, model: Model, var):
        return model.translate(var).x

    def get_search_progress_log(self):
        return self.model.search_progress_log.log
