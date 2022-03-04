import pytest
import numpy as np
import pandas as pd

from trees.RobustTree import RobustTreeClassifier


def test_RobustTree_X_noninteger_error():
    """Test whether X is integer-valued"""

    clf = RobustTreeClassifier(depth=1, time_limit=20)

    with pytest.raises(
        ValueError,
        match="Found non-integer values.",
    ):
        data = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0.1, 1], "y": [1, 1, -1, -1, -1]}
        )
        y = data.pop("y")
        clf.fit(data, y)


def test_RobustTree_cost_shape_error():
    """Test whether X and cost have the same size and columns"""
    clf = RobustTreeClassifier(depth=1, time_limit=20)
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
        clf.fit(data, y, costs=costs, budget=5, verbose=False)

    # Different number of features
    with pytest.raises(
        ValueError,
        match="Input covariates has 2 columns but uncertainty costs has 3 columns",
    ):
        costs = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 7, 1], "x3": [1, 1, 1, 1, 1]},
            index=["A", "B", "C", "D", "E"],
        )
        clf.fit(data, y, costs=costs, budget=5, verbose=False)

    # Different column names
    with pytest.raises(
        KeyError,
        match="uncertainty costs should have the same columns as the input covariates",
    ):
        costs = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x3": [1, 2, 1, 7, 1]},
            index=["A", "B", "C", "D", "E"],
        )
        clf.fit(data, y, costs=costs, budget=5, verbose=False)

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
        clf.fit(data_np, y, costs=costs, budget=5, verbose=False)

    # When X is a dataframe, but costs are not
    with pytest.raises(
        TypeError,
        match="uncertainty costs should be a Pandas DataFrame with the same columns as the input covariates",
    ):
        costs = np.transpose([[1, 2, 2, 2, 3], [1, 2, 1, 7, 1]])
        clf.fit(data, y, costs=costs, budget=5, verbose=False)


@pytest.mark.test_gurobi
def test_RobustTree_prediction_shape_error():
    """Test whether X and cost have the same size and columns"""
    # Run some quick model that finishes in 1 second
    clf = RobustTreeClassifier(depth=1, time_limit=20)
    train = pd.DataFrame(
        {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0, 1], "y": [1, 1, -1, -1, -1]},
        index=["A", "B", "C", "D", "E"],
    )
    y = train.pop("y")
    clf.fit(train, y, verbose=False)

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
        clf.fit(train_nodf, y, verbose=False)
        clf.predict(test)


@pytest.mark.test_gurobi
def test_RobustTree_with_uncertainty_success():
    clf = RobustTreeClassifier(depth=1, time_limit=20)
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
    clf.fit(train, y, costs=costs, budget=5, verbose=False)
    assert hasattr(clf, "model")

    y_pred = clf.predict(test)
    assert y_pred.shape[0] == test.shape[0]


@pytest.mark.test_gurobi
def test_RobustTree_no_uncertainty_success():
    clf = RobustTreeClassifier(depth=1, time_limit=20)
    train = pd.DataFrame(
        {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0, 1], "y": [1, 1, -1, -1, -1]},
        index=["A", "B", "C", "D", "E"],
    )
    test = pd.DataFrame(
        {"x1": [1, 2, 2, 2], "x2": [1, 2, 1, 7]}, index=["F", "G", "H", "I"]
    )
    y = train.pop("y")
    clf.fit(train, y, verbose=False)
    assert hasattr(clf, "model")

    y_pred = clf.predict(test)
    assert y_pred.shape[0] == test.shape[0]
