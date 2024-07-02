import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


class Binarizer(BaseEstimator, TransformerMixin):
    """
    A transformer that binarizes categorical, integer, and real-valued columns.

    This transformer follows the scikit-learn fit-transform paradigm and can be used
    in scikit-learn pipelines.

    Parameters
    ----------
    categorical_cols : list, optional
        List of categorical column names to be one-hot encoded.
    integer_cols : list, optional
        List of integer column names to be binarized.
    real_cols : list, optional
        List of real-valued column names to be discretized and then binarized.
    n_bins : int, default=4
        The number of bins to use for discretizing real-valued columns.
    bin_strategy : {'uniform', 'quantile'}, default='uniform'
        The strategy to use for binning real-valued columns.
        'uniform': All bins in each feature have identical widths.
        'quantile': All bins in each feature have the same number of points.

    Attributes
    ----------
    encoders_ : dict
        Dictionary of fitted encoders for each column type.
    column_names_ : list
        List of column names in the transformed output.

    Examples
    --------
    >>> import pandas as pd
    >>> from odtlearn.utils.binarizer import Binarizer
    >>> df = pd.DataFrame({
    ...     'cat': ['A', 'B', 'C', 'A'],
    ...     'int': [1, 2, 3, 2],
    ...     'real': [0.1, 0.5, 0.7, 0.9]
    ... })
    >>> binarizer = Binarizer(
    ...     categorical_cols=['cat'],
    ...     integer_cols=['int'],
    ...     real_cols=['real'],
    ...     n_bins=2
    ... )
    >>> X_bin = binarizer.fit_transform(df)
    """

    def __init__(
        self,
        categorical_cols=None,
        integer_cols=None,
        real_cols=None,
        n_bins=4,
        bin_strategy="uniform",
    ):
        assert any(
            [x is not None for x in [categorical_cols, integer_cols, real_cols]]
        ), (
            "Must provide at least one of the three options of a list of categorical columns "
            "or integer columns or real valued columns to binarize."
        )

        if len(real_cols) > 0 and n_bins is None:
            raise ValueError(
                "The number of bins must be provided for encoding real columns."
            )
        if len(real_cols) > 0 and bin_strategy is None:
            raise ValueError(
                "The bin strategy must be provided for encoding real columns."
            )
        if (
            len(real_cols) > 0
            and bin_strategy is None
            or bin_strategy not in ["uniform", "quantile"]
        ):
            raise ValueError(
                "The bin strategy must be one of the following: 'uniform' or 'quantile'."
            )

        self.categorical_cols = categorical_cols or []
        self.integer_cols = integer_cols or []
        self.real_cols = real_cols or []
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.encoders_ = {}
        self.column_names_ = None

    def fit(self, X, y=None):
        """
        Fit the Binarizer to the input data.

        This method learns the encoding schemes for categorical, integer, and real-valued columns.

        Parameters
        ----------
        X : array-like or pandas DataFrame of shape (n_samples, n_features)
            The input samples to be binarized.
        y : None
            Ignored. This parameter exists only for compatibility with sklearn.

        Returns
        -------
        self : object
            Returns self.
        """
        X = pd.DataFrame(X)

        if self.categorical_cols:
            cat_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="if_binary",
            )
            cat_encoder.fit(X[self.categorical_cols])
            self.encoders_["categorical"] = cat_encoder

        if self.integer_cols:
            int_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            int_encoder.fit(X[self.integer_cols])
            self.encoders_["integer"] = int_encoder

        if self.real_cols:
            real_encoder = KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy=self.bin_strategy
            )
            real_encoder.fit(X[self.real_cols])
            self.encoders_["real"] = real_encoder

        # Generate column names for the transformed data
        self.column_names_ = self._get_feature_names_out()

        return self

    def transform(self, X):
        """
        Transform the input data using the fitted Binarizer.

        Parameters
        ----------
        X : array-like or pandas DataFrame of shape (n_samples, n_features)
            The input samples to be binarized.

        Returns
        -------
        X_bin : pandas DataFrame
            The binarized input data.
        """
        X = pd.DataFrame(X)
        result = pd.DataFrame()

        if self.categorical_cols:
            cat_bin = self.encoders_["categorical"].transform(X[self.categorical_cols])
            result = pd.concat(
                [
                    result,
                    pd.DataFrame(
                        cat_bin,
                        columns=self.encoders_["categorical"].get_feature_names_out(
                            self.categorical_cols
                        ),
                    ),
                ],
                axis=1,
            )

        if self.integer_cols:
            int_bin = self.encoders_["integer"].transform(X[self.integer_cols])
            int_features = self.encoders_["integer"].get_feature_names_out(
                self.integer_cols
            )
            for col in self.integer_cols:
                col_features = [f for f in int_features if f.startswith(f"{col}_")]
                col_bin = int_bin[
                    :, [i for i, f in enumerate(int_features) if f in col_features]
                ]
                col_bin_cumsum = np.cumsum(col_bin, axis=1)
                result = pd.concat(
                    [result, pd.DataFrame(col_bin_cumsum, columns=col_features)], axis=1
                )

        if self.real_cols:
            real_bin = self.encoders_["real"].transform(X[self.real_cols])
            real_bin_df = pd.DataFrame(real_bin, columns=self.real_cols)
            for col in self.real_cols:
                col_bin = pd.get_dummies(real_bin_df[col], prefix=col, dtype=np.float64)
                col_bin_cumsum = col_bin.cumsum(axis=1)
                col_bin_cumsum.columns = [
                    x.replace(".0", "") for x in col_bin_cumsum.columns
                ]
                result = pd.concat([result, col_bin_cumsum], axis=1)

        for col in self.column_names_:
            if col not in result.columns:
                result[col] = 0

        return result[self.column_names_]

    def _get_feature_names_out(self):
        """Get feature names for the binarized columns."""
        feature_names = []

        if self.categorical_cols:
            feature_names.extend(
                self.encoders_["categorical"].get_feature_names_out(
                    self.categorical_cols
                )
            )

        if self.integer_cols:
            int_features = self.encoders_["integer"].get_feature_names_out(
                self.integer_cols
            )
            feature_names.extend(int_features)

        if self.real_cols:
            for col in self.real_cols:
                feature_names.extend([f"{col}_{i}" for i in range(self.n_bins)])

        return feature_names


def binarize(
    df, categorical_cols, integer_cols, real_cols, n_bins=4, bin_strategy="uniform"
):
    """
    Parameters
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

    Return
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
