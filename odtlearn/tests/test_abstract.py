from abc import ABCMeta

from odtlearn.opt_dt import OptimalDecisionTree


def test_dt_abc():
    OptimalDecisionTree.__abstractmethods__ = set()

    test_opt_dt = OptimalDecisionTree(
        solver="cbc", depth=10, time_limit=60, num_threads=5, verbose=True
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
