import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from odtlearn.flow_oct import BendersOCT, FlowOCT


# fmt: off
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
def synthetic_data_2():
    """
    X2              |
    |               |
    1    + - -      |    -
    |               |
    |---------------|-------------
    |               |
    0    - - - +    |    - - -
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
            [0, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])

    return X, y


# fmt: on


@pytest.fixture
def small_binary_dataset():
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(20, 5))
    y = np.random.randint(0, 2, size=20)
    return X, y


def test_FlowOCT_X_nonbinary_error():
    # Test that we raise a ValueError if X matrix has values other than zero or one
    clf = FlowOCT(solver="cbc", depth=1, time_limit=2, _lambda=1)

    with pytest.raises(
        AssertionError,
        match="Expecting all values of covariate matrix to be either 0 or 1",
    ):
        X = np.arange(10).reshape(10, 1)
        y = np.random.randint(2, size=X.shape[0])
        clf.fit(X, y)

    with pytest.raises(
        ValueError,
        match=r"Found columns .* that contain values other than 0 or 1.",
    ):
        data = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0, 1], "y": [1, 1, -1, -1, -1]}
        )
        y = data.pop("y")
        clf.fit(data, y)


# Test that we raise an error if X and y have different number of rows
def test_FlowOCT_X_data_shape_error():
    X = np.ones(100).reshape(100, 1)

    clf = FlowOCT(solver="cbc", depth=1, time_limit=2, _lambda=1)

    with pytest.raises(
        ValueError, match="Found input variables with inconsistent numbers of samples"
    ):
        y_diff_size = np.random.randint(2, size=X.shape[0] + 1)
        clf.fit(X, y_diff_size)


# test that tree is fitted before trying to fit, predict, print, or plot
def test_check_fit(synthetic_data_1):
    X, y = synthetic_data_1
    clf = FlowOCT(solver="cbc", depth=1, time_limit=2, _lambda=1)
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


# Test that if we are given a pandas dataframe, we keep the original data and its labels
def test_FlowOCT_classifier():
    train = pd.DataFrame(
        {"x1": [1, 0, 0, 0, 1], "x2": [1, 1, 1, 0, 1], "y": [1, 1, 0, 0, 0]},
        index=["A", "B", "C", "D", "E"],
    )
    y = train.pop("y")
    test = pd.DataFrame({"x1": [1, 1, 0, 0, 1], "x2": [1, 1, 1, 0, 1]})
    clf = FlowOCT(solver="cbc", depth=1, time_limit=20, _lambda=0.2)

    clf.fit(train, y)
    # Test that after running the fit method we have b, w, and p
    assert hasattr(clf, "b_value")
    assert hasattr(clf, "w_value")
    assert hasattr(clf, "p_value")

    y_pred = clf.predict(test)
    assert y_pred.shape == (train.shape[0],)


@pytest.mark.parametrize(
    "d, _lambda, benders, expected_pred, solver",
    [
        (
            0,
            0,
            False,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "gurobi",
        ),
        (
            1,
            0,
            False,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            "gurobi",
        ),
        (
            2,
            0,
            False,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]),
            "gurobi",
        ),
        (
            0,
            0,
            True,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "gurobi",
        ),
        (
            1,
            0,
            True,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            "gurobi",
        ),
        (
            2,
            0,
            True,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]),
            "gurobi",
        ),
        (
            2,
            0.51,
            False,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
            "gurobi",
        ),
        (
            2,
            0.51,
            True,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
            "gurobi",
        ),
        (
            0,
            0,
            False,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "cbc",
        ),
        (
            1,
            0,
            False,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            "cbc",
        ),
        (
            2,
            0,
            False,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]),
            "cbc",
        ),
        (
            0,
            0,
            True,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "cbc",
        ),
        (
            1,
            0,
            True,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]),
            "cbc",
        ),
        (
            2,
            0,
            True,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]),
            "cbc",
        ),
        (
            2,
            0.51,
            False,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
            "cbc",
        ),
        (
            2,
            0.51,
            True,
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
            "cbc",
        ),
    ],
)
def test_FlowOCT_same_predictions(
    synthetic_data_1, d, _lambda, benders, expected_pred, solver, skip_solver
):
    if skip_solver:
        pytest.skip(reason="Gurobi license not available.")
    X, y = synthetic_data_1

    if benders:
        stcl = BendersOCT(
            solver=solver,
            depth=d,
            time_limit=100,
            _lambda=_lambda,
            num_threads=None,
            obj_mode="acc",
        )
        stcl.fit(X, y)
        # stcl.print_tree()
        assert_allclose(stcl.predict(X), expected_pred)

    else:
        stcl = FlowOCT(
            solver=solver,
            depth=d,
            time_limit=100,
            _lambda=_lambda,
            num_threads=None,
            obj_mode="acc",
        )
        stcl.fit(X, y)
        # stcl.print_tree()
        assert_allclose(stcl.predict(X), expected_pred)


@pytest.mark.parametrize(
    "benders, obj_mode ,expected_pred, solver",
    [
        (
            False,
            "acc",
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "gurobi",
        ),
        (
            False,
            "balance",
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            "gurobi",
        ),
        (
            True,
            "acc",
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "gurobi",
        ),
        (
            True,
            "balance",
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            "gurobi",
        ),
        (
            False,
            "acc",
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "cbc",
        ),
        (
            False,
            "balance",
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            "cbc",
        ),
        (
            True,
            "acc",
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "cbc",
        ),
        (
            True,
            "balance",
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            "cbc",
        ),
    ],
)
def test_FlowOCT_obj_mode(
    synthetic_data_2, benders, obj_mode, expected_pred, solver, skip_solver
):
    X, y = synthetic_data_2

    if skip_solver:
        pytest.skip(reason="Testing on github actions")

    if benders:
        bstcl = BendersOCT(
            solver=solver,
            depth=2,
            time_limit=100,
            _lambda=0,
            num_threads=None,
            obj_mode=obj_mode,
        )
        bstcl.fit(X, y)
        # stcl.print_tree()
        assert_allclose(bstcl.predict(X), expected_pred)
    else:
        stcl = FlowOCT(
            solver=solver,
            depth=2,
            time_limit=100,
            _lambda=0,
            num_threads=None,
            obj_mode=obj_mode,
        )
        stcl.fit(X, y)
        # stcl.print_tree()
        assert_allclose(stcl.predict(X), expected_pred)


def test_FlowOCT_plot_print(synthetic_data_1):
    X, y = synthetic_data_1
    stcl = FlowOCT(
        solver="cbc",
        depth=1,
        time_limit=100,
        num_threads=None,
        obj_mode="acc",
    )
    stcl.fit(X, y)
    stcl.print_tree()
    stcl.plot_tree()


def test_wrong_objective_FlowOCT(synthetic_data_1):
    X, y = synthetic_data_1
    with pytest.raises(
        ValueError,
        match="objective must be one of 'acc', 'balance', or 'custom'",
    ):
        stcl = FlowOCT(
            solver="cbc",
            depth=1,
            time_limit=100,
            num_threads=None,
            obj_mode="sdafasdf",
        )
        stcl.fit(X, y)

    with pytest.raises(
        ValueError,
        match="objective must be one of 'acc', 'balance', or 'custom'",
    ):
        bstcl = BendersOCT(
            solver="cbc",
            depth=1,
            time_limit=100,
            num_threads=None,
            obj_mode="aaa",
        )
        bstcl.fit(X, y)


@pytest.mark.parametrize("OCTClass", [FlowOCT, BendersOCT])
def test_custom_weights(OCTClass, small_binary_dataset):
    X, y = small_binary_dataset

    # Create custom weights that heavily favor class 1
    weights = np.ones_like(y)
    weights[y == 1] = 10

    # Fit the model with custom weights
    oct = OCTClass(solver="cbc", obj_mode="custom", depth=2, time_limit=10)
    oct.fit(X, y, weights=weights)
    y_pred = oct.predict(X)

    # The prediction should favor class 1 due to higher weights
    assert np.mean(y_pred) > np.mean(y)


@pytest.mark.parametrize("OCTClass", [FlowOCT, BendersOCT])
def test_custom_weights_ignored_warning(OCTClass, small_binary_dataset):
    X, y = small_binary_dataset
    weights = np.ones_like(y)

    oct = OCTClass(solver="cbc", obj_mode="acc", depth=2, time_limit=10)
    with pytest.warns(
        UserWarning, match="Weights are ignored because obj_mode is not 'custom'."
    ):
        oct.fit(X, y, weights=weights)


@pytest.mark.parametrize("OCTClass", [FlowOCT, BendersOCT])
def test_custom_weights_missing_error(OCTClass, small_binary_dataset):
    X, y = small_binary_dataset

    oct = OCTClass(solver="cbc", obj_mode="custom", depth=2, time_limit=10)
    with pytest.raises(
        ValueError, match="Weights must be provided when obj_mode is 'custom'."
    ):
        oct.fit(X, y)


@pytest.mark.parametrize("OCTClass", [FlowOCT, BendersOCT])
def test_custom_weights_wrong_length(OCTClass, small_binary_dataset):
    X, y = small_binary_dataset
    weights = np.ones(len(y) - 1)  # Wrong length

    oct = OCTClass(solver="cbc", obj_mode="custom", depth=2, time_limit=10)
    with pytest.raises(
        ValueError, match="The number of weights must match the number of samples."
    ):
        oct.fit(X, y, weights=weights)


@pytest.mark.parametrize("OCTClass", [FlowOCT, BendersOCT])
def test_weight_consistency(OCTClass, small_binary_dataset):
    X, y = small_binary_dataset

    # Fit with 'acc' mode
    oct_acc = OCTClass(solver="cbc", obj_mode="acc", depth=2, time_limit=10)
    oct_acc.fit(X, y)

    # Fit with 'custom' mode and uniform weights
    oct_custom = OCTClass(solver="cbc", obj_mode="custom", depth=2, time_limit=10)
    oct_custom.fit(X, y, weights=np.ones_like(y))

    # Predictions should be the same
    assert np.array_equal(oct_acc.predict(X), oct_custom.predict(X))
