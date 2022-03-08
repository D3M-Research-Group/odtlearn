import pytest
import numpy as np
import pandas as pd

from trees.PrescriptiveTree import PrescriptiveTreeClassifier


@pytest.fixture
def data():
    df = pd.read_csv("../../data/prescriptive_tree/train_50.csv")
    return df


# Test that we raise a ValueError if X matrix has values other than zero or one
def test_PrescriptiveTree_X_nonbinary_error():

    clf = PrescriptiveTreeClassifier(depth=1, time_limit=300, method="IPW")

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
def test_PrescriptiveTree_X_data_shape_error():
    X = np.ones(10).reshape(10, 1)

    clf = PrescriptiveTreeClassifier(depth=1, time_limit=300, method="IPW")

    with pytest.raises(
        ValueError, match="Found input variables with inconsistent numbers of samples"
    ):
        t = np.random.randint(2, size=X.shape[0] + 1)
        y = np.random.rand(X.shape[0] + 1)
        ipw = np.random.rand(X.shape[0] + 1)
        # y_diff_size = np.random.randint(2, size=X.shape[0] + 1)
        clf.fit(X, t, y, ipw)


# Test that we raise an error if IPW and y_hat are not in correct format
def test_PrescriptiveTree_X_helpers_error(data):
    X = np.ones(10).reshape(10, 1)
    t = np.random.randint(2, size=X.shape[0])
    y = np.random.rand(10)

    # what if we pass None for different permutations of IPW, DM, DR
    with pytest.raises(
        AssertionError, match="Inverse propensity weights cannot be None"
    ):
        clf = PrescriptiveTreeClassifier(depth=1, time_limit=300, method="IPW")
        clf.fit(X, t, y)

    with pytest.raises(AssertionError, match="Counterfactual estimates cannot be None"):
        clf = PrescriptiveTreeClassifier(depth=1, time_limit=300, method="DM")
        clf.fit(X, t, y)

    # what if we pass ipw outside of range (0, 1]
    with pytest.raises(
        AssertionError, match=r"Inverse propensity weights must be in the range \(0, 1]"
    ):
        ipw = np.random.rand(10) + 1
        clf = PrescriptiveTreeClassifier(depth=1, time_limit=300, method="IPW")
        clf.fit(X, t, y, ipw)

    # what if we pass y_hat with columns that don't match up to # of treatments
    with pytest.raises(
        AssertionError,
        match=r"Found counterfactual estimates for .*",
    ):
        # df = pd.read_csv("../../data/prescriptive_tree/train_50.csv")
        df = data
        y_hat = df[["lasso0", "lasso1", "lasso1"]]
        clf = PrescriptiveTreeClassifier(depth=1, time_limit=300, method="DM")
        clf.fit(X=X, t=t, y=y, y_hat=y_hat)


# Test that we raise an error if t isn't discrete and starts from 0
def test_PrescriptiveTree_X_treatment_error():
    clf = PrescriptiveTreeClassifier(depth=1, time_limit=300, method="IPW")

    with pytest.raises(
        AssertionError,
        match="The set of treatments must be discrete starting from {0, 1, ...}",
    ):
        X = np.ones(10).reshape(10, 1)
        t = np.random.randint(low=1, high=3, size=X.shape[0])
        y = np.random.rand(10)
        ipw = np.random.rand(10)
        clf.fit(X, t, y, ipw)


@pytest.mark.test_gurobi
# Test that if we are given a pandas dataframe, we keep the original data and its labels
def test_PrescriptiveTree_classifier(data):
    # df = pd.read_csv("../../data/prescriptive_tree/train_50.csv")
    df = data
    clf = PrescriptiveTreeClassifier(depth=1, time_limit=300, method="DR")

    X = df.iloc[:, :20]
    t = df["t"]
    y = df["y"]
    ipw = df["prob_t_pred_tree"]
    y_hat = df[["linear0", "linear1"]]

    clf.fit(X, t, y, ipw, y_hat)
    # Test that after running the fit method we have b, w, and p
    assert hasattr(clf, "X_")
    assert hasattr(clf, "t_")
    assert hasattr(clf, "y_")
    assert hasattr(clf, "b_value")
    assert hasattr(clf, "w_value")
    assert hasattr(clf, "p_value")

    t_pred = clf.predict(X)
    assert len(t_pred) == X.shape[0]
