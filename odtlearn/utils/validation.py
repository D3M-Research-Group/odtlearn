import numpy as np
import pandas as pd
from sklearn.utils.validation import (
    _assert_all_finite,
    check_array,
    check_consistent_length,
    column_or_1d,
)


def check_ipw(X, ipw):
    """
    This function checks the propensity weights and counterfactual predictions

    Parameters
    ----------
    X: The input/training data
    ipw: A vector or array-like object for inverse propensity weights. Only needed when running IPW/DR

    Returns
    -------
    The converted version of ipw after passing the series of checks
    """
    if ipw is not None:
        ipw = column_or_1d(ipw, warn=True)

        if ipw.dtype.kind == "O":
            ipw = ipw.astype(np.float64)

        assert (
            min(ipw) > 0 and max(ipw) <= 1
        ), "Inverse propensity weights must be in the range (0, 1]"

        check_consistent_length(X, ipw)
    return ipw


def check_y_hat(X, treatments, y_hat):
    """
    This function checks the propensity weights and counterfactual predictions

    Parameters
    ----------
    X: The input/training data
    treatments: A vector of the unique treatment values in the dataset.
    y_hat: A multi-dimensional array-like object for counterfactual predictions. Only needed when running DM/DR

    Returns
    -------
    The converted versions of ipw and y_hat after passing the series of checks
    """
    if y_hat is not None:
        y_hat = check_array(y_hat)

        # y_hat has to have as many columns as there are treatments

        assert y_hat.shape[1] == len(
            treatments
        ), f"Found counterfactual estimates for {y_hat.shape[1]} treatments. \
        There are {len(treatments)} unique treatments in the data"

        check_consistent_length(X, y_hat)
    else:
        assert y_hat is not None, "Counterfactual estimates cannot be None"

    return y_hat


def check_y(X, y):
    """
    This function checks the shape and contents of the observed outcomes

    Parameters
    ----------
    X: The input/training data
    y: A vector or array-like object for the observed outcomes corresponding to treatment t

    Returns
    -------
    The converted version of y after passing the series of checks
    """
    # check consistent length
    y = column_or_1d(y, warn=True)
    _assert_all_finite(y)
    if y.dtype.kind == "O":
        y = y.astype(np.float64)

    check_consistent_length(X, y)
    return y


def check_columns_match(original_columns, new_data):
    """
    Check that the column names of a new data frame match the column names used when used to fit the model

    Parameters
    ----------
    original_columns: List of column names from the data set used to fit the model
    new_data: The numpy matrix or pd dataframe new data set for
    which we want to make predictions

    Returns
    -------
    ValueError if column names do not match, otherwise None
    """

    if isinstance(new_data, pd.DataFrame):
        new_column_names = new_data.columns
        # take difference of sets
        non_matched_columns = set(new_column_names) - set(original_columns)
        if len(non_matched_columns) > 0:
            raise ValueError(
                f"Columns {list(non_matched_columns)} found in prediction data, but not found in fit data."
            )
    else:
        # we are assuming the order of columns matches and we will just check that shapes match
        # assuming here that new_data is a numpy matrix
        assert (
            len(original_columns) != new_data.shape[0]
        ), f"Fit data has {len(original_columns)} columns but new data has {new_data.shape[0]} columns."


def check_binary(df):
    # TO-DO: truncate output if lots of non_binary_columns
    if isinstance(df, pd.DataFrame):
        non_binary_columns = [
            col for col in df if not np.isin(df[col].dropna().unique(), [0, 1]).all()
        ]
        if len(non_binary_columns) > 0:
            raise ValueError(
                f"Found columns ({non_binary_columns}) that contain values other than 0 or 1."
            )
    else:
        assert (
            (df == 0) | (df == 1)
        ).all(), "Expecting all values of covariate matrix to be either 0 or 1."


def check_integer(df):
    if not np.array_equal(df.values, df.values.astype(int)):
        raise ValueError("Found non-integer values.")


def check_same_as_X(X, X_col_labels, G, G_label):
    """Check if a DataFrame G has the columns of X"""
    # Check if X has shape of G
    if X.shape[1] != G.shape[1]:
        raise ValueError(
            f"Input covariates has {X.shape[1]} columns but {G_label} has {G.shape[1]} columns"
        )

    # Check if X has same columns as G
    if isinstance(G, pd.DataFrame):
        if not np.array_equal(np.sort(X_col_labels), np.sort(G.columns)):
            raise KeyError(
                f"{G_label} should have the same columns as the input covariates"
            )
        return G
    else:
        # Check if X has default column labels or not
        # if not np.array_equal(X_col_labels, np.arange(0, G.shape[1])):
        if not np.array_equal(
            X_col_labels, np.array([f"X_{i}" for i in np.arange(0, G.shape[1])])
        ):
            raise TypeError(
                f"{G_label} should be a Pandas DataFrame with the same columns as the input covariates"
            )
        return pd.DataFrame(
            G, columns=np.array([f"X_{i}" for i in np.arange(0, G.shape[1])])
        )
