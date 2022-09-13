import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose


from odtlearn.RobustTree import RobustTreeClassifier


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


@pytest.fixture
def synthetic_data_2():
    X = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])

    return X, y


@pytest.fixture
def synthetic_costs_2():
    """Uncertainty in 5 points at [0,0] on X1 can cause it to flip
    to [1,0] if needed to misclassify
    Uncertainty in 1 point at [1,1] on X2 can cause it to flip
    to another value if needed to misclassify
    All other points certain
    """
    costs = np.array(
        [
            [1, 4, 4, 4],
            [1, 4, 4, 4],
            [1, 4, 4, 4],
            [1, 4, 4, 4],
            [1, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 1, 1, 1],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ]
    )
    return costs


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
    assert hasattr(clf, "grb_model")

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
    assert hasattr(clf, "grb_model")

    y_pred = clf.predict(test)
    assert y_pred.shape[0] == test.shape[0]


@pytest.mark.parametrize(
    "d, expected_pred",
    [
        (0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        (1, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])),
        (2, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])),
    ],
)
def test_RobustTree_correctness(synthetic_data_1, d, expected_pred):
    X, y = synthetic_data_1
    robust_classifier = RobustTreeClassifier(
        depth=d,
        time_limit=100,
    )

    robust_classifier.fit(X, y, verbose=False)
    assert_allclose(robust_classifier.predict(X), expected_pred)


@pytest.mark.parametrize(
    "d, budget, expected_pred",
    [
        (0, 2, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        (1, 2, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])),
        (2, 2, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])),
        (2, 5, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])),
    ],
)
def test_RobustTree_uncertainty_correctness(
    synthetic_data_1, synthetic_costs_1, d, budget, expected_pred
):
    """
    Scenario 0: Root assigns 0
    Scenario 1: Split on X2 (X1 causes too many worst-case misclassifications)
    Scenario 2: Perfect split (uncertainty budget not large enough)
    Scenario 3: Split X2, split X1 at node 3 but assign 0 at node 2 (because uncertainty in X1)
    """
    X, y = synthetic_data_1
    costs = synthetic_costs_1
    robust_classifier = RobustTreeClassifier(
        depth=d,
        time_limit=100,
    )
    robust_classifier.fit(X, y, costs=costs, budget=budget, verbose=False)
    assert_allclose(robust_classifier.predict(X), expected_pred)


@pytest.mark.parametrize(
    "d, budget",
    [
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 5),
    ],
)
@pytest.mark.test_gurobi
def test_RobustTree_categoricals_success(
    synthetic_data_2, synthetic_costs_2, d, budget
):
    X, y = synthetic_data_2
    X = pd.DataFrame(X, columns=["x1", "x2.1", "x2.2", "x2.3"])
    costs = pd.DataFrame(synthetic_costs_2, columns=["x1", "x2.1", "x2.2", "x2.3"])
    robust_classifier = RobustTreeClassifier(
        depth=d,
        time_limit=100,
    )
    categories = {"x2": ["x2.1", "x2.2", "x2.3"]}
    robust_classifier.fit(
        X,
        y,
        costs=costs,
        budget=budget,
        categories=categories,
        verbose=False,
    )
    assert hasattr(robust_classifier, "grb_model")
