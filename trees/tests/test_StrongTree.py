import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from trees.StrongTree import StrongTreeClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


# Test that we raise a ValueError if X matrix has values other than zero or one
def test_StrongTree_X_nonbinary_error():

    clf = StrongTreeClassifier(depth=1, time_limit=2, _lambda=1)

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
def test_StrongTree_X_data_shape_error():
    X = np.ones(100).reshape(100, 1)

    clf = StrongTreeClassifier(depth=1, time_limit=2, _lambda=1)

    with pytest.raises(
        ValueError, match="Found input variables with inconsistent numbers of samples"
    ):
        y_diff_size = np.random.randint(2, size=X.shape[0] + 1)
        clf.fit(X, y_diff_size)


# Test that if we are given a pandas dataframe, we keep the original data and its labels
def test_StrongTree_classifier():
    X = np.random.randint(2, size=(10, 3))
    y = np.random.randint(2, size=X.shape[0])
    clf = StrongTreeClassifier(depth=1, time_limit=2, _lambda=1)

    clf.fit(X, y)
    # Test that after running the fit method we have b, w, and p
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")
    assert hasattr(clf, "b_value")
    assert hasattr(clf, "w_value")
    assert hasattr(clf, "p_value")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


# # Test that on toy data we get the same predictions as our replication code
# def test_StrongTree_same_predictions(data):
#     predictions = []  # toy data predictions
#     X, y = data
#     X_test = []
#     clf = StrongTreeClassifier(depth=1, time_limit=2, _lambda=1)
#     clf.fit(X, y)

#     assert_allclose(X_test, predictions)
