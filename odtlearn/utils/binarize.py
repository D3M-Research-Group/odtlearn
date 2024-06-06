import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def binarize(
    df, categorical_cols, integer_cols, real_cols, n_bins=4, bin_strategy="uniform"
):
    """
    parameters
    ----------
    df: pandas dataframe
        A dataframe with only categorical/integer columns. There should not be any NA values.
    categorical_cols: list
        A list consisting of the names of categorical columns of df
    integer_cols: list
        A list consisting of the names of integer columns of df
    real_cols: list
        A list consisting of the names of real columns of df
    n_bins: int
        The number of bins to be used for encoding real columns
    bin_strategy: str
        The strategy to be used for discretizing real columns. It can be one of the following:
        'uniform': All bins in each feature have identical widths.
        'quantile': All bins in each feature have the same number of points.

    return
    ----------
    the binarized version of the input dataframe.

    This function encodes each categorical column as a one-hot vector, i.e.,
    for each level of the feature, it creates a new binary column with a value
    of one if and only if the original column has the corresponding level.
    A similar approach for encoding integer features is used with a slight change.
    The new binary column should have a value of one if and only if the main column
    has the corresponding value or any value smaller than it.
    We first discretize the real columns according to the number of bins and the discretization strategy
    and then treat them as integer columns.
    """

    assert len(categorical_cols) > 0 or len(integer_cols) > 0 or len(real_cols) > 0, (
        "Must provide at least one of the three options of a list of categorical columns "
        "or integer columns or real valued columns to binarize."
    )

    if len(real_cols) > 0 and n_bins is None:
        raise ValueError(
            "The number of bins must be provided for encoding real columns."
        )
    if len(real_cols) > 0 and bin_strategy is None:
        raise ValueError("The bin strategy must be provided for encoding real columns.")
    if (
        len(real_cols) > 0
        and bin_strategy is None
        or bin_strategy not in ["uniform", "quantile"]
    ):
        raise ValueError(
            "The bin strategy must be one of the following: 'uniform' or 'quantile'."
        )

    if len(categorical_cols) > 0:
        X_cat = np.array(df[categorical_cols])
        enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
        X_cat_enc = enc.fit_transform(X_cat).toarray()
        categorical_cols_enc = enc.get_feature_names_out(categorical_cols)
        X_cat_enc = X_cat_enc.astype(int)
    if len(real_cols) > 0:
        # We first bucketize the real columns according to number of bins and the
        # discretization strategy and then treat them as integer columns
        discretizer_unif = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy=bin_strategy
        )
        df[real_cols] = discretizer_unif.fit_transform(df[real_cols])
        integer_cols = integer_cols + real_cols
    if len(integer_cols) > 0:
        X_int = np.array(df[integer_cols])
        enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
        X_int_enc = enc.fit_transform(X_int).toarray()
        integer_cols_enc = enc.get_feature_names_out(integer_cols)
        X_int_enc = X_int_enc.astype(int)

        for col in integer_cols:
            col_enc_set = []
            col_offset = None
            for i, col_enc in enumerate(integer_cols_enc):
                if col in col_enc:
                    col_enc_set.append(col_enc)
                    if col_offset is None:
                        col_offset = i
            if len(col_enc_set) < 3:
                continue
            for i, col_enc in enumerate(col_enc_set):
                if i == 0:
                    continue
                X_int_enc[:, (col_offset + i)] = (
                    X_int_enc[:, (col_offset + i)] | X_int_enc[:, (col_offset + i - 1)]
                )
    if len(categorical_cols) > 0 and len(integer_cols) > 0:
        df_enc = pd.DataFrame(
            np.c_[X_cat_enc, X_int_enc],
            columns=list(categorical_cols_enc) + list(integer_cols_enc),
        )
    elif len(categorical_cols) > 0 and len(integer_cols) == 0:
        df_enc = pd.DataFrame(
            X_cat_enc,
            columns=list(categorical_cols_enc),
        )
    elif len(categorical_cols) == 0 and len(integer_cols) > 0:
        df_enc = pd.DataFrame(
            X_int_enc,
            columns=list(integer_cols_enc),
        )
    return df_enc
