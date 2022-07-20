import numpy as np
import pandas as pd


def check_columns_match(original_columns, new_data):
    """
    :param original_columns: List of column names from the data set used to fit the model
    :param new_data: The numpy matrix or pd dataframe new data set for
    which we want to make predictions
    :return ValueError if column names do not match, otherwise None
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
