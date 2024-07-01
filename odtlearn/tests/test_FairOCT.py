from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from odtlearn.fair_oct import (
    FairCSPOCT,
    FairEOddsOCT,
    FairEOppOCT,
    FairOCT,
    FairPEOCT,
    FairSPOCT,
)
from odtlearn.flow_oct import FlowOCT


# @pytest.fixture
def class_names():
    return [
        FairCSPOCT,
        FairEOddsOCT,
        FairEOppOCT,
        # FairOCT,
        FairPEOCT,
        FairSPOCT,
    ]


# @pytest.fixture
def solvers():
    return ["gurobi", "cbc"]


# @pytest.fixture
def objectives():
    return ["acc", "balance"]


# @pytest.fixture
def obj_solver_class_combo():
    return list(product(*[objectives(), solvers(), class_names()]))


# @pytest.fixture
def obj_class_combo():
    return list(product(*[objectives(), class_names()]))


# fmt: off
@pytest.fixture
def synthetic_data_1():
    """
    This is the data we generate in this function
       X2                    |
       |                     |
       1    5W: 4(-) 1(+)    |     2W: 1(-) 1(+)
       |    2B: 2(-)         |     5B: 3(-) 2(+)
       |                     |
       |                     |
       |---------------------|------------------------
       |                     |
       0    4W: 3(-) 1(+)    |         3W: 1(-) 2(+)
       |    1B:      1(+)    |         6B: 1(-) 5(+)
       |                     |
       |___________0_________|__________1_____________X1
    """
    X = np.array(
        [
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0],
            [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
            [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1],
            [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1], [0, 1]
        ]
    )
    protect_feat = np.array(
        [
            0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
        ]
    )
    y = np.array(
        [
            0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
            1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
        ]
    )

    protect_feat = protect_feat.reshape(-1, 1)
    # X = np.concatenate((X, P), axis=1)
    legit_factor = X[:, 1]

    return X, y, protect_feat, legit_factor


# fmt: on
@pytest.mark.parametrize(
    "obj_mode, solver, fair_class",
    obj_solver_class_combo(),
    # [("acc", "gurobi"), ("balance", "gurobi"), ("acc", "cbc"), ("balance", "cbc")],
)
def test_FairOCT_same_predictions(
    synthetic_data_1, obj_mode, solver, fair_class, skip_solver
):
    if skip_solver:
        pytest.skip(reason="Testing on github actions")
    X, y, protect_feat, legit_factor = synthetic_data_1

    fcl = fair_class(
        solver=solver,
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_bound=1,
        num_threads=None,
        obj_mode=obj_mode,
    )

    fcl_dep = FairOCT(
        solver=solver,
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_type=None,
        fairness_bound=1,
        num_threads=None,
        obj_mode=obj_mode,
    )

    stcl = FlowOCT(
        solver=solver,
        depth=2,
        time_limit=100,
        _lambda=0,
        num_threads=None,
        obj_mode=obj_mode,
    )

    stcl.fit(X, y)
    stcl_pred = stcl.predict(X)

    fcl.fit(X, y, protect_feat, legit_factor)
    fcl_pred = fcl.predict(X)

    fcl_dep.fit(X, y, protect_feat, legit_factor)
    fcl_dep_pred = fcl_dep.predict(X)

    assert_allclose(fcl_pred, stcl_pred)
    assert_allclose(fcl_dep_pred, stcl_pred)


@pytest.mark.parametrize(
    "f, b, g0_value, solver, class_name",
    [
        ("SP", 1, 0.214, "gurobi", FairSPOCT),
        ("SP", 0.2, 0.5, "gurobi", FairSPOCT),
        ("PE", 1, 0.111, "gurobi", FairPEOCT),
        ("PE", 0.04, 0, "gurobi", FairPEOCT),
        ("SP", 1, 0.214, "cbc", FairSPOCT),
        ("SP", 0.2, 0.5, "cbc", FairSPOCT),
        ("PE", 1, 0.111, "cbc", FairPEOCT),
        ("PE", 0.04, 0, "cbc", FairPEOCT),
    ],
)
def test_FairOCT_metrics(
    synthetic_data_1, f, b, g0_value, solver, class_name, skip_solver
):
    if skip_solver:
        pytest.skip(reason="Testing on github actions")

    X, y, protect_feat, legit_factor = synthetic_data_1

    fcl = class_name(
        solver=solver,
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_bound=b,
        num_threads=None,
        obj_mode="acc",
    )

    fcl_dep = FairOCT(
        solver=solver,
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_type=f,
        fairness_bound=b,
        num_threads=None,
        obj_mode="acc",
    )

    fcl.fit(X, y, protect_feat, legit_factor)
    fcl_dep.fit(X, y, protect_feat, legit_factor)
    if f == "SP":
        sp_val_dep = fcl_dep.get_SP(protect_feat, fcl.predict(X))
        sp_val = fcl.calc_metric(protect_feat, fcl.predict(X))
        assert_allclose(np.round(sp_val[(0, 1)], 3), g0_value)
        assert_allclose(np.round(sp_val_dep[(0, 1)], 3), g0_value)
    else:
        eq_val_dep = fcl_dep.get_EqOdds(protect_feat, y, fcl.predict(X))
        eq_val = fcl.calc_metric(protect_feat, y, fcl.predict(X))
        assert_allclose(np.round(eq_val[(0, 0, 1)], 3), g0_value)
        assert_allclose(np.round(eq_val_dep[(0, 0, 1)], 3), g0_value)


@pytest.mark.parametrize(
    "obj_mode, class_name",
    obj_class_combo(),
)
# test that tree is fitted before trying to fit, predict, print, or plot
def test_check_fit(synthetic_data_1, obj_mode, class_name):
    X, _, _, _ = synthetic_data_1

    # check first for depreciated class
    fcl = FairOCT(
        solver="cbc",
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_type="SP",
        fairness_bound=1,
        num_threads=None,
        obj_mode=obj_mode,
    )
    with pytest.raises(
        NotFittedError,
        match=(
            f"This {fcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        fcl.predict(X)

    with pytest.raises(
        NotFittedError,
        match=(
            f"This {fcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        fcl.print_tree()

    with pytest.raises(
        NotFittedError,
        match=(
            f"This {fcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        fcl.plot_tree()

    # now do for new version of classes
    fcl = class_name(
        solver="cbc",
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_bound=1,
        num_threads=None,
        obj_mode=obj_mode,
    )
    with pytest.raises(
        NotFittedError,
        match=(
            f"This {fcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        fcl.predict(X)

    with pytest.raises(
        NotFittedError,
        match=(
            f"This {fcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        fcl.print_tree()

    with pytest.raises(
        NotFittedError,
        match=(
            f"This {fcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        fcl.plot_tree()


@pytest.mark.parametrize(
    "class_name",
    class_names(),
)
def test_FairOCT_visualize_tree(synthetic_data_1, class_name):
    X, y, protect_feat, legit_factor = synthetic_data_1

    # first test depreciated version
    fcl_dep = FairOCT(
        solver="cbc",
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_type="SP",
        fairness_bound=1,
        num_threads=None,
        obj_mode="acc",
    )

    fcl_dep.fit(X, y, protect_feat, legit_factor)
    fcl_dep.print_tree()
    fcl_dep.plot_tree()

    # now test each class
    fcl = class_name(
        solver="cbc",
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_bound=1,
        num_threads=None,
        obj_mode="acc",
    )

    fcl.fit(X, y, protect_feat, legit_factor)
    fcl.print_tree()
    fcl.plot_tree()


@pytest.mark.parametrize(
    "f, pd_data, class_name",
    [
        ("SP", True, FairSPOCT),
        ("SP", False, FairSPOCT),
        ("PE", True, FairPEOCT),
        ("PE", False, FairPEOCT),
        ("CSP", True, FairCSPOCT),
        ("CSP", False, FairCSPOCT),
        ("CPE", True, FairEOddsOCT),
        ("CPE", False, FairEOddsOCT),
    ],
)
# test that we can properly handle data passed as pd.DataFrame
def test_handle_pandas_cols(synthetic_data_1, f, pd_data, class_name):
    X, y, protect_feat, legit_factor = synthetic_data_1
    fcl = FairOCT(
        solver="cbc",
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_type=f,
        fairness_bound=1,
        num_threads=None,
        obj_mode="acc",
    )
    if pd_data:
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        protect_feat = pd.DataFrame(protect_feat)

    fcl.fit(X, y, protect_feat, legit_factor)

    if f == "SP":
        metric_val = fcl.get_SP(protect_feat, fcl.predict(X))
        assert len(metric_val.keys()) == 4
    elif f == "CSP":
        metric_val = fcl.get_CSP(protect_feat, legit_factor, fcl.predict(X))
        assert len(metric_val.keys()) == 8
    elif f == "PE":
        metric_val = fcl.get_EqOdds(protect_feat, y, fcl.predict(X))
        assert len(metric_val.keys()) == 8
    else:
        metric_val = fcl.get_CondEqOdds(protect_feat, legit_factor, y, fcl.predict(X))
        assert len(metric_val.keys()) == 16


@pytest.mark.parametrize(
    "class_name",
    class_names(),
)
# test that assertion error is thrown when invalid objective mode is given
def test_bad_obj_mode(synthetic_data_1, class_name):
    X, y, protect_feat, legit_factor = synthetic_data_1
    with pytest.raises(
        ValueError,
        match="objective must be one of 'acc', 'balance', or 'custom'",
    ):
        fcl = class_name(
            solver="cbc",
            positive_class=1,
            depth=2,
            _lambda=0,
            time_limit=100,
            fairness_bound=1,
            num_threads=None,
            obj_mode="inacc",
        )
        fcl.fit(X, y, protect_feat, legit_factor)


# test that assertion error is thrown when invalid objective mode is given
def test_bad_obj_mode_dep(synthetic_data_1):
    X, y, protect_feat, legit_factor = synthetic_data_1
    with pytest.raises(
        ValueError,
        match="Invalid objective mode. obj_mode should be one of acc or balance.",
    ):
        fcl = FairOCT(
            solver="cbc",
            positive_class=1,
            depth=2,
            _lambda=0,
            time_limit=100,
            fairness_type="SP",
            fairness_bound=1,
            num_threads=None,
            obj_mode="inacc",
        )
        fcl.fit(X, y, protect_feat, legit_factor)


@pytest.mark.parametrize(
    "f, b",
    [("SP", 1), ("CSP", 1), ("PE", 1), ("CPE", 1)],
)
def test_fairness_metric_summary(synthetic_data_1, f, b):
    X, y, protect_feat, legit_factor = synthetic_data_1
    fcl = FairOCT(
        solver="cbc",
        positive_class=1,
        depth=2,
        _lambda=0.01,
        time_limit=100,
        fairness_type=f,
        fairness_bound=b,
        num_threads=None,
        obj_mode="acc",
        verbose=False,
    )
    fcl.fit(X, y, protect_feat, legit_factor)

    fcl.fairness_metric_summary(f, fcl.predict(X))


@pytest.mark.parametrize(
    "fair_class", [FairSPOCT, FairCSPOCT, FairPEOCT, FairEOppOCT, FairEOddsOCT]
)
def test_custom_weights_error(synthetic_data_1, fair_class):
    X, y, protect_feat, legit_factor = synthetic_data_1
    weights = np.array([0.1, 0.2, 0.3])  # Incorrect number of weights

    model = fair_class(
        solver="cbc",
        positive_class=1,
        depth=2,
        _lambda=0,
        obj_mode="custom",
        fairness_bound=1,
    )

    with pytest.raises(
        ValueError, match="The number of weights must match the number of samples."
    ):
        model.fit(X, y, protect_feat, legit_factor, weights=weights)


@pytest.mark.parametrize(
    "fair_class", [FairSPOCT, FairCSPOCT, FairPEOCT, FairEOppOCT, FairEOddsOCT]
)
def test_custom_weights_missing(synthetic_data_1, fair_class):
    X, y, protect_feat, legit_factor = synthetic_data_1

    model = fair_class(
        solver="cbc",
        positive_class=1,
        depth=2,
        _lambda=0,
        obj_mode="custom",
        fairness_bound=1,
    )

    with pytest.raises(
        ValueError, match="Weights must be provided when obj_mode is 'custom'."
    ):
        model.fit(X, y, protect_feat, legit_factor)
