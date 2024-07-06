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
    Check and validate inverse probability weights (IPW).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    ipw : array-like of shape (n_samples,)
        The inverse probability weights to be checked.

    Returns
    -------
    ipw : ndarray of shape (n_samples,)
        The validated and potentially converted inverse probability weights.

    Raises
    ------
    ValueError
        If ipw has inconsistent number of samples with X.
    AssertionError
        If any value in ipw is not in the range (0, 1].


    Examples
    --------
    >>> import numpy as np
    >>> from odtlearn.utils.validation import check_ipw
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> ipw = np.array([0.5, 0.7, 0.3])
    >>> validated_ipw = check_ipw(X, ipw)
    >>> print(validated_ipw)
    [0.5 0.7 0.3]
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
    Check and validate counterfactual predictions (y_hat).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    treatments : array-like
        The unique treatment values.
    y_hat : array-like of shape (n_samples, n_treatments)
        The counterfactual predictions to be checked.

    Returns
    -------
    y_hat : ndarray of shape (n_samples, n_treatments)
        The validated and potentially converted counterfactual predictions.

    Raises
    ------
    ValueError
        If y_hat has inconsistent dimensions with X or treatments.
    AssertionError
        If y_hat is None.


    Examples
    --------
    >>> import numpy as np
    >>> from odtlearn.utils.validation import check_y_hat
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> treatments = [0, 1]
    >>> y_hat = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> validated_y_hat = check_y_hat(X, treatments, y_hat)
    >>> print(validated_y_hat)
    [[0.1 0.2]
     [0.3 0.4]
     [0.5 0.6]]
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
    Check and validate target values (y).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values to be checked.

    Returns
    -------
    y : ndarray of shape (n_samples,)
        The validated and potentially converted target values.

    Raises
    ------
    ValueError
        If y has inconsistent number of samples with X.

    Examples
    --------
    >>> import numpy as np
    >>> from odtlearn.utils.validation import check_y
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([0, 1, 0])
    >>> validated_y = check_y(X, y)
    >>> print(validated_y)
    [0. 1. 0.]
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
    Check if the columns in new_data match the original_columns.

    Parameters
    ----------
    original_columns : list
        The list of column names from the original data.
    new_data : array-like or pandas.DataFrame
        The new data to be checked.

    Returns
    -------
    bool
        True if the columns match, False otherwise.

    Raises
    ------
    ValueError
        If new_data is a DataFrame and contains columns not present in original_columns.
    AssertionError
        If new_data is not a DataFrame and has a different number of columns than original_columns.

    Notes
    -----
    This function performs different checks based on whether new_data is a pandas DataFrame or not:
    - For DataFrames: It checks if all columns in new_data are present in original_columns.
    - For non-DataFrames: It checks if the number of columns matches the length of original_columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from odtlearn.utils.validation import check_columns_match
    >>> original_cols = ['A', 'B', 'C']
    >>> new_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    >>> result = check_columns_match(original_cols, new_data)
    >>> print(result)
    True
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
            return True
    else:
        # we are assuming the order of columns matches and we will just check that shapes match
        # assuming here that new_data is a numpy matrix
        assert (
            len(original_columns) == new_data.shape[1]
        ), f"Fit data has {len(original_columns)} columns but new data has {new_data.shape[1]} columns."


def check_binary(df):
    """
    Check if all values in the DataFrame are binary (0 or 1).

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        The data to be checked.

    Raises
    ------
    ValueError
        If df is a DataFrame and contains columns with non-binary values.
    AssertionError
        If df is not a DataFrame and contains non-binary values.

    Notes
    -----
    This function performs different checks based on whether df is a pandas DataFrame or not:
    - For DataFrames: It identifies columns containing non-binary values.
    - For non-DataFrames: It checks if all values are either 0 or 1.

    Examples
    --------
    >>> import pandas as pd
    >>> from odtlearn.utils.validation import check_binary
    >>> df = pd.DataFrame({'A': [0, 1, 0], 'B': [1, 1, 0]})
    >>> check_binary(df)  # This will not raise an error
    >>> df['C'] = [0, 1, 2]
    >>> check_binary(df)  # This will raise a ValueError
    ValueError: Found columns (['C']) that contain values other than 0 or 1.
    """
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
    """
    Check if all values in the DataFrame are integers.

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        The data to be checked.

    Raises
    ------
    ValueError
        If df contains non-integer values.

    Examples
    --------
    >>> import pandas as pd
    >>> from odtlearn.utils.validation import check_integer
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> check_integer(df)  # This will not raise an error
    >>> df['C'] = [1.5, 2.0, 3.0]
    >>> check_integer(df)  # This will raise a ValueError
    ValueError: Found non-integer values.
    """
    if not np.array_equal(df.values, df.values.astype(int)):
        raise ValueError("Found non-integer values.")


def check_same_as_X(X, X_col_labels, G, G_label):
    """
    Check if a DataFrame G has the same structure as X.

    Parameters
    ----------
    X : pandas.DataFrame
        The reference DataFrame.
    X_col_labels : array-like
        The column labels of X.
    G : pandas.DataFrame or array-like
        The DataFrame or array to be checked against X.
    G_label : str
        A label for G to be used in error messages.

    Returns
    -------
    pandas.DataFrame
        G converted to a DataFrame if it wasn't already.

    Raises
    ------
    ValueError
        If G has a different number of columns than X.
    KeyError
        If G is a DataFrame and its columns don't match X_col_labels.
    TypeError
        If G is not a DataFrame and X has non-default column labels.


    Examples
    --------
    >>> import pandas as pd
    >>> from odtlearn.utils.validation import check_same_as_X
    >>> X = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> G = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    >>> result = check_same_as_X(X, X.columns, G, 'Test DataFrame')
    >>> print(result)
       A  B
    0  5  7
    1  6  8
    """
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
