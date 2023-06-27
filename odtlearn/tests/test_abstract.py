from abc import ABCMeta

from odtlearn.opt_dt import OptimalDecisionTree
from odtlearn.utils.solver import Solver


def test_dt_abc():
    OptimalDecisionTree.__abstractmethods__ = set()

    test_opt_dt = OptimalDecisionTree(
        solver="gurobi", depth=10, time_limit=60, num_threads=5, verbose=True
    )
    vars_defined = test_opt_dt._define_variables()
    constraints_defined = test_opt_dt._define_constraints()
    objective_defined = test_opt_dt._define_objective()
    main_problem_made = test_opt_dt._create_main_problem()
    fit_on_data = test_opt_dt.fit()
    predict_on_data = test_opt_dt.predict()
    assert isinstance(OptimalDecisionTree, ABCMeta)
    assert vars_defined is None
    assert constraints_defined is None
    assert objective_defined is None
    assert main_problem_made is None
    assert fit_on_data is None
    assert predict_on_data is None


def test_solver_abc():
    Solver.__abstractmethods__ = set()
    test_solver = Solver("test")
    add_constrs = test_solver.add_constrs()
    add_vars = test_solver.add_vars()
    get_var_value = test_solver.get_var_value()
    lin_expr = test_solver.lin_expr()
    quick_sum = test_solver.quicksum()
    opt = test_solver.optimize()
    set_obj = test_solver.set_objective()
    assert isinstance(Solver, ABCMeta)
    assert add_constrs is None
    assert add_vars is None
    assert get_var_value is None
    assert lin_expr is None
    assert quick_sum is None
    assert opt is None
    assert set_obj is None
