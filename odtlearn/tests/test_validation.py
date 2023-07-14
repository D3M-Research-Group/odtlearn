import numpy as np
import pandas as pd
import pytest

from odtlearn.utils.validation import (
    check_columns_match,
    check_ipw,
    check_y,
    check_y_hat,
)


def test_check_ipw_dtype():
    X = np.arange(10).reshape(10, 1)
    ipw = np.random.rand(10)
    ipw = ipw.astype(np.dtype("O"))
    ipw_new = check_ipw(X, ipw.ravel())
    assert ipw_new.dtype == np.float64


def test_check_ipw_range():
    X = np.arange(10).reshape(10, 1)
    ipw = np.arange(10).reshape(10, 1)
    with pytest.raises(
        AssertionError, match=r"Inverse propensity weights must be in the range \(0, 1]"
    ):
        check_ipw(X, ipw.ravel())


def test_check_ipw_None():
    X = np.arange(10).reshape(10, 1)
    ipw_new = check_ipw(X, None)
    assert ipw_new is None


def test_check_y_hat():
    X = np.arange(10).reshape(10, 1)
    t = np.random.randint(2, size=X.shape[0])
    with pytest.raises(AssertionError, match="Counterfactual estimates cannot be None"):
        check_y_hat(X, t, None)


def test_check_y_dtype():
    X = np.arange(10).reshape(10, 1)
    y = np.random.rand(10)
    y = y.astype(np.dtype("O"))
    y_new = check_y(X, y)
    assert y_new.dtype == np.float64


def test_check_columns_match_pandas():
    original_cols = [f"X{i}" for i in range(10)]
    X_new = pd.DataFrame({f"X{i}": np.random.rand(10) for i in range(11)})
    print(X_new.columns)
    with pytest.raises(
        ValueError,
        match="Columns (.*) found in prediction data, but not found in fit data.",
    ):
        check_columns_match(original_cols, X_new)


def test_check_columns_match_pandas_names():
    original_cols = [f"X{i}" for i in range(10)]
    X_new = pd.DataFrame({f"X{i}": np.random.rand(10) for i in range(10)})
    assert check_columns_match(original_cols, X_new)
