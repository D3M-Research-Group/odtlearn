import numpy as np
import pandas as pd
import pkg_resources
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from odtlearn.flow_opt import FlowOPT_DM, FlowOPT_DR, FlowOPT_IPW


@pytest.fixture
def data():
    data_stream = pkg_resources.resource_stream(
        "odtlearn", "data/prescriptive/train_50.csv"
    )
    df = pd.read_csv(data_stream)
    return df


# Test that we raise a ValueError if X matrix has values other than zero or one
def test_FlowOPT_X_nonbinary_error():

    clf = FlowOPT_IPW(solver="cbc", depth=1, time_limit=300)

    with pytest.raises(
        AssertionError,
        match="Expecting all values of covariate matrix to be either 0 or 1",
    ):
        X = np.arange(10).reshape(10, 1)
        t = np.random.randint(2, size=X.shape[0])
        y = np.random.rand(10)
        ipw = np.random.rand(10)

        clf.fit(X, t, y, ipw)


# Test that we raise an error if X and y have different number of rows
def test_FlowOPT_X_data_shape_error():
    X = np.ones(10).reshape(10, 1)

    clf = FlowOPT_IPW(solver="cbc", depth=1, time_limit=300)

    with pytest.raises(
        ValueError, match="Found input variables with inconsistent numbers of samples"
    ):
        t = np.random.randint(2, size=X.shape[0] + 1)
        y = np.random.rand(X.shape[0] + 1)
        ipw = np.random.rand(X.shape[0] + 1)
        # y_diff_size = np.random.randint(2, size=X.shape[0] + 1)
        clf.fit(X, t, y, ipw)


# Test that we raise an error if IPW and y_hat are not in correct format
def test_FlowOPT_X_helpers_error(data):
    X = np.ones(10).reshape(10, 1)
    t = np.random.randint(2, size=X.shape[0])
    y = np.random.rand(10)

    # what if we pass ipw outside of range (0, 1]
    with pytest.raises(
        AssertionError, match=r"Inverse propensity weights must be in the range \(0, 1]"
    ):
        ipw = np.random.rand(10) + 1
        clf = FlowOPT_IPW(solver="cbc", depth=1, time_limit=300)
        clf.fit(X, t, y, ipw)

    # what if we pass y_hat with columns that don't match up to # of treatments
    with pytest.raises(
        AssertionError,
        match=r"Found counterfactual estimates for .*",
    ):
        # df = pd.read_csv("../../data/prescriptive_tree/train_50.csv")
        df = data
        y_hat = df[["lasso0", "lasso1", "lasso1"]]
        clf = FlowOPT_DM(solver="cbc", depth=1, time_limit=300)
        clf.fit(X=X, t=t, y=y, y_hat=y_hat)


# Test that we raise an error if t isn't discrete and starts from 0
def test_FlowOPT_X_treatment_error():
    clf = FlowOPT_IPW(solver="cbc", depth=1, time_limit=300)

    with pytest.raises(
        AssertionError,
        match="The set of treatments must be discrete starting from {0, 1, ...}",
    ):
        X = np.ones(10).reshape(10, 1)
        t = np.random.randint(low=1, high=3, size=X.shape[0])
        y = np.random.rand(10)
        ipw = np.random.rand(10)
        clf.fit(X, t, y, ipw)


# fmt: off
@pytest.mark.parametrize("method,solver, expected_pred", [
    ('DR', 'gurobi', np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
                               1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
                               1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0])),
    ('DM', 'gurobi', np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
                               1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1,
                               1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0])),
    ('IPW', 'gurobi', np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
                                1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                0, 1])),
    ('DR', 'cbc', np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
                            1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
                            1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0])),
    ('DM', 'cbc', np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
                            1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1,
                            1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0])),
    ('IPW', 'cbc', np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
                            1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                            0, 1]))
])
# fmt: on
def test_FlowOPT_classifier(data, method, solver, expected_pred, skip_solver):
    if skip_solver:
        pytest.skip(reason="Testing on github actions")
    df = data
    X = df.iloc[:, :20]
    t = df["t"]
    y = df["y"]
    ipw = df["prob_t_pred_tree"]
    y_hat = df[["linear0", "linear1"]]

    if method == "DR":
        clf = FlowOPT_DR(solver=solver, depth=2, time_limit=300)
        clf.fit(X, t, y, ipw, y_hat)
    elif method == "DM":
        clf = FlowOPT_DM(solver=solver, depth=2, time_limit=300)
        clf.fit(X, t, y, y_hat)
    else:
        clf = FlowOPT_IPW(solver=solver, depth=2, time_limit=300)
        clf.fit(X, t, y, ipw)

    # Test that after running the fit method we have b, w, and p
    assert hasattr(clf, "b_value")
    assert hasattr(clf, "w_value")
    assert hasattr(clf, "p_value")

    assert_allclose(clf.predict(X), expected_pred)


@pytest.mark.parametrize("method", ["DR", "DM", "IPW"])
# test that tree is fitted before trying to fit, predict, print, or plot
def test_check_fit(data, method):
    df = data
    X = df.iloc[:, :20]
    if method == "DR":
        clf = FlowOPT_DR(solver="cbc", depth=2, time_limit=300)
    elif method == "DM":
        clf = FlowOPT_DM(solver="cbc", depth=2, time_limit=300)
    else:
        clf = FlowOPT_IPW(solver="cbc", depth=2, time_limit=300)
    with pytest.raises(
        NotFittedError,
        match=(
            f"This {clf.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        clf.predict(X)

    with pytest.raises(
        NotFittedError,
        match=(
            f"This {clf.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        clf.print_tree()

    with pytest.raises(
        NotFittedError,
        match=(
            f"This {clf.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        clf.plot_tree()


@pytest.mark.parametrize("method", ["DR", "DM", "IPW"])
def test_FairOCT_visualize_tree(data, method):
    df = data
    X = df.iloc[:, :20]
    t = df["t"]
    y = df["y"]
    ipw = df["prob_t_pred_tree"]
    y_hat = df[["linear0", "linear1"]]
    if method == "DR":
        clf = FlowOPT_DR(solver="cbc", depth=2, time_limit=300)
        clf.fit(X, t, y, ipw, y_hat)
    elif method == "DM":
        clf = FlowOPT_DM(solver="cbc", depth=2, time_limit=300)
        clf.fit(X, t, y, y_hat)
    else:
        clf = FlowOPT_IPW(solver="cbc", depth=2, time_limit=300)
        clf.fit(X, t, y, ipw)
    clf.print_tree()
    clf.plot_tree()
