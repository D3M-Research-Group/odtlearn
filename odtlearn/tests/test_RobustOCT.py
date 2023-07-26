import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from odtlearn.robust_oct import RobustOCT


@pytest.fixture
def synthetic_data_1():
    """
    X2              |
    |               |
    1    + +        |    -
    |               |
    |---------------|-------------
    |               |
    0    - - - -    |    + + +
    |    - - -      |
    |______0________|_______1_______X1
    """
    X = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 1],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])

    return X, y


@pytest.fixture
def synthetic_costs_1():
    """Uncertainty in 5 points at [0,0] on X1 can cause it to flip
    to [1,0] if needed to misclassify
    Uncertainty in 1 point at [1,1] on X2 can cause it to flip
    to [1,0] if needed to misclassify
    All other points certain
    """
    costs = np.array(
        [
            [1, 4],
            [1, 4],
            [1, 4],
            [1, 4],
            [1, 4],
            [4, 4],
            [4, 4],
            [4, 4],
            [4, 4],
            [4, 4],
            [4, 1],
            [4, 4],
            [4, 4],
        ]
    )
    return costs


def test_RobustOCT_X_noninteger_error():
    """Test whether X is integer-valued"""
    clf = RobustOCT(solver="cbc", depth=1, time_limit=20)

    with pytest.raises(
        ValueError,
        match="Found non-integer values.",
    ):
        data = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0.1, 1], "y": [1, 1, -1, -1, -1]}
        )
        y = data.pop("y")
        clf.fit(data, y)


def test_RobustOCT_cost_shape_error():
    """Test whether X and cost have the same size and columns"""
    clf = RobustOCT(solver="cbc", depth=1, time_limit=20)
    data = pd.DataFrame(
        {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0, 1], "y": [1, 1, -1, -1, -1]},
        index=["A", "B", "C", "D", "E"],
    )
    y = data.pop("y")

    # Different number of data samples
    with pytest.raises(
        ValueError,
        match="Input covariates has 5 samples, but uncertainty costs has 4",
    ):
        costs = pd.DataFrame(
            {"x1": [1, 2, 2, 2], "x2": [1, 2, 1, 1]}, index=["A", "B", "C", "D"]
        )
        clf.fit(data, y, costs=costs, budget=5)

    # Different number of features
    with pytest.raises(
        ValueError,
        match="Input covariates has 2 columns but uncertainty costs has 3 columns",
    ):
        costs = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 7, 1], "x3": [1, 1, 1, 1, 1]},
            index=["A", "B", "C", "D", "E"],
        )
        clf.fit(data, y, costs=costs, budget=5)

    # Different column names
    with pytest.raises(
        KeyError,
        match="uncertainty costs should have the same columns as the input covariates",
    ):
        costs = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x3": [1, 2, 1, 7, 1]},
            index=["A", "B", "C", "D", "E"],
        )
        clf.fit(data, y, costs=costs, budget=5)

    # When X is not a dataframe, but costs is a dataframe with column names
    with pytest.raises(
        KeyError,
        match="uncertainty costs should have the same columns as the input covariates",
    ):
        data_np = np.array([[1, 2, 2, 2, 3], [1, 2, 1, 0, 1]]).transpose()
        costs = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 7, 1]},
            index=["A", "B", "C", "D", "E"],
        )
        clf.fit(data_np, y, costs=costs, budget=5)

    # When X is a dataframe, but costs are not
    with pytest.raises(
        TypeError,
        match="uncertainty costs should be a Pandas DataFrame with the same columns as the input covariates",
    ):
        costs = np.transpose([[1, 2, 2, 2, 3], [1, 2, 1, 7, 1]])
        clf.fit(data, y, costs=costs, budget=5)


def test_RobustOCT_prediction_shape_error():
    """Test whether X and cost have the same size and columns"""
    # Run some quick model that finishes in 1 second
    clf = RobustOCT(solver="cbc", depth=1, time_limit=20)
    train = pd.DataFrame(
        {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0, 1], "y": [1, 1, -1, -1, -1]},
        index=["A", "B", "C", "D", "E"],
    )
    y = train.pop("y")
    clf.fit(train, y)

    # Non-integer data
    with pytest.raises(
        ValueError,
        match="Found non-integer values.",
    ):
        test = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0.1, 1]},
            index=["F", "G", "H", "I", "J"],
        )
        clf.predict(test)

    # Different number of features
    with pytest.raises(
        ValueError,
        match="Input covariates has 2 columns but test covariates has 3 columns",
    ):
        test = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 7, 1], "x3": [1, 1, 1, 1, 1]},
            index=["F", "G", "H", "I", "J"],
        )
        clf.predict(test)

    # Different column names
    with pytest.raises(
        KeyError,
        match="test covariates should have the same columns as the input covariates",
    ):
        test = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x3": [1, 2, 1, 7, 1]},
            index=["F", "G", "H", "I", "J"],
        )
        clf.predict(test)

    # When X is a dataframe, but test is not
    with pytest.raises(
        TypeError,
        match="test covariates should be a Pandas DataFrame with the same columns as the input covariates",
    ):
        test = np.transpose([[1, 2, 2, 2, 3], [1, 2, 1, 7, 1]])
        clf.predict(test)

    # When X is not a dataframe, but test is a dataframe with column names
    with pytest.raises(
        KeyError,
        match="test covariates should have the same columns as the input covariates",
    ):
        test = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 7, 1]},
            index=["F", "G", "H", "I", "J"],
        )
        train_nodf = np.transpose([[1, 2, 2, 2, 3], [1, 2, 1, 0, 1]])
        clf = RobustOCT(solver="cbc", depth=1, time_limit=20)
        clf.fit(train_nodf, y)
        clf.predict(test)


def test_RobustOCT_with_uncertainty_success(skip_solver):
    if skip_solver:
        pytest.skip(reason="Testing on github actions")
    clf = RobustOCT(solver="cbc", depth=1, time_limit=20)
    train = pd.DataFrame(
        {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0, 1], "y": [1, 1, -1, -1, -1]},
        index=["A", "B", "C", "D", "E"],
    )
    test = pd.DataFrame(
        {"x1": [1, 2, 2, 2], "x2": [1, 2, 1, 7]}, index=["F", "G", "H", "I"]
    )
    y = train.pop("y")
    costs = pd.DataFrame(
        {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 7, 1]}, index=["A", "B", "C", "D", "E"]
    )
    clf.fit(train, y, costs=costs, budget=5)
    assert hasattr(clf, "b_value")
    assert hasattr(clf, "w_value")

    y_pred = clf.predict(test)
    assert y_pred.shape[0] == test.shape[0]


def test_RobustOCT_no_uncertainty_success(skip_solver):
    if skip_solver:
        pytest.skip(reason="Testing on github actions")
    clf = RobustOCT(solver="cbc", depth=1, time_limit=20)
    train = pd.DataFrame(
        {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0, 1], "y": [1, 1, -1, -1, -1]},
        index=["A", "B", "C", "D", "E"],
    )
    test = pd.DataFrame(
        {"x1": [1, 2, 2, 2], "x2": [1, 2, 1, 7]}, index=["F", "G", "H", "I"]
    )
    y = train.pop("y")
    clf.fit(train, y)
    assert hasattr(clf, "b_value")
    assert hasattr(clf, "w_value")

    y_pred = clf.predict(test)
    assert y_pred.shape[0] == test.shape[0]


@pytest.mark.parametrize(
    "d, expected_pred, solver",
    [
        (
            0,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "gurobi",
        ),
        (
            1,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            "gurobi",
        ),
        (
            2,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]),
            "gurobi",
        ),
        (0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), "cbc"),
        (1, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]), "cbc"),
        (2, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]), "cbc"),
    ],
)
def test_RobustOCT_correctness(synthetic_data_1, d, expected_pred, solver, skip_solver):
    if skip_solver:
        pytest.skip(reason="Testing on github actions")
    X, y = synthetic_data_1
    robust_classifier = RobustOCT(
        solver=solver,
        depth=d,
        time_limit=100,
    )

    robust_classifier.fit(X, y)
    assert_allclose(robust_classifier.predict(X), expected_pred)


@pytest.mark.parametrize(
    "d, budget, expected_pred, solver",
    [
        (
            0,
            2,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "gurobi",
        ),
        (
            1,
            2,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            "gurobi",
        ),
        (
            2,
            2,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]),
            "gurobi",
        ),
        (
            2,
            5,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            "gurobi",
        ),
        (
            0,
            2,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "cbc",
        ),
        (
            1,
            2,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            "cbc",
        ),
        (
            2,
            2,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]),
            "cbc",
        ),  # slow
        (
            2,
            5,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            "cbc",
        ),  # slow
    ],
)
def test_RobustOCT_uncertainty_correctness(
    synthetic_data_1,
    synthetic_costs_1,
    d,
    budget,
    expected_pred,
    solver,
    skip_solver,
):
    """
    Scenario 0: Root assigns 0
    Scenario 1: Split on X2 (X1 causes too many worst-case misclassifications)
    Scenario 2: Perfect split (uncertainty budget not large enough)
    Scenario 3: Split X2, split X1 at node 3 but assign 0 at node 2 (because uncertainty in X1)
    """
    if skip_solver:
        pytest.skip(reason="Testing on github actions")
    X, y = synthetic_data_1
    costs = synthetic_costs_1
    robust_classifier = RobustOCT(
        solver=solver,
        depth=d,
        time_limit=100,
    )
    robust_classifier.fit(X, y, costs=costs, budget=budget)
    assert_allclose(robust_classifier.predict(X), expected_pred)


# test that tree is fitted before trying to fit, predict, print, or plot
def test_check_fit(synthetic_data_1):
    X, y = synthetic_data_1
    rcl = RobustOCT(
        solver="cbc",
        depth=1,
        time_limit=100,
    )
    with pytest.raises(
        NotFittedError,
        match=(
            f"This {rcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        rcl.predict(X)

    with pytest.raises(
        NotFittedError,
        match=(
            f"This {rcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        rcl.print_tree()

    with pytest.raises(
        NotFittedError,
        match=(
            f"This {rcl.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            f"appropriate arguments before using this estimator."
        ),
    ):
        rcl.plot_tree()


def test_RobustOCT_visualize_tree(synthetic_data_1, synthetic_costs_1, skip_solver):
    if skip_solver:
        pytest.skip(reason="Testing on github actions")
    X, y = synthetic_data_1
    costs = synthetic_costs_1
    robust_classifier = RobustOCT(
        solver="cbc",
        depth=1,
        time_limit=100,
    )
    robust_classifier.fit(X, y, costs=costs, budget=2)

    robust_classifier.print_tree()
    robust_classifier.plot_tree()
