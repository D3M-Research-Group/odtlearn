import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import time
from trees.utils.StrongTreeUtils import (
    check_columns_match,
    check_binary
)

# Include Tree.py, FlowOCT.py and BendersOCT.py in StrongTrees folder
from trees.utils.Tree import Tree
from trees.utils.StrongTreeFairOCT import FairOCT


class FairTreeClassifier(ClassifierMixin, BaseEstimator):
    """ Description of this estimator here


    Parameters
    ----------
    depth : int, default=1
        A parameter specifying the depth of the tree
    time_limit : int
        Add description here
    _lambda : int
        Add description here

    Examples
    --------

    """

    def __init__(self, depth, time_limit, _lambda, positive_class,
                 fairness_type = None, fairness_bound = 1, num_threads=1):
        # this is where we will initialize the values we want users to provide
        self.depth = depth
        self.time_limit = time_limit
        self._lambda = _lambda
        self.num_threads = num_threads

        self.fairness_type = fairness_type
        self.fairness_bound = fairness_bound
        self.positive_class = positive_class

        self.X_col_labels = None
        self.X_col_dtypes = None
        self.y_dtypes = None

        self.P_col_labels = None
        self.P_col_dtypes = None

        self.L_col_labels = None
        self.L_col_dtypes = None


    def extract_metadata(self, X, y, P, L):
        """A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API

        """
        if isinstance(X, pd.DataFrame):
            self.X_col_labels = X.columns
            self.X_col_dtypes = X.dtypes
        else:
            self.X_col_labels = np.array([f'X_{i}' for i in np.arange(0, X.shape[1])])


        if isinstance(P, pd.DataFrame):
            self.P_col_labels = P.columns
            self.P_col_dtypes = P.dtypes
        else:
            self.P_col_labels = np.array([f'P_{i}' for i in np.arange(0, P.shape[1])])


        if isinstance(L, pd.DataFrame):
            self.L_col_labels = L.columns
            self.L_col_dtypes = L.dtypes
        else:
            self.L_col_labels = np.array([f'L_{i}' for i in np.arange(0, L.shape[1])])



        self.labels = np.unique(y)


    def fit(self, X, y, P, L):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        P : array-like, shape (n_samples,) or (n_samples, n_p)
            The protected feature columns (Race, gender, etc)

        L : array-like, shape (n_samples,) or (n_samples, n_l)
            The legitimate factor columns (e.g., prior number of criminal acts)

        Returns
        -------
        self : object
            Returns self.
        """
        # store column information and dtypes if any
        self.extract_metadata(X, y, P, L)
        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)
        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)



        'Here we need to convert P and L to np.arrays. We need a function.' \
        'We also need to check that P.shape and L.shape is not (n_sample, )'
        P = np.array(P)
        L = np.array(L)

        # keep original data
        self.X_ = X
        self.y_ = y

        # Instantiate tree object here
        self.tree = Tree(self.depth)

        self.primal = FairOCT(
            X,
            y,
            self.tree,
            self.X_col_labels,
            self.labels,
            self._lambda,
            self.time_limit,
            self.num_threads,
            self.fairness_type,
            self.fairness_bound,
            self.positive_class,
            P,
            self.P_col_labels,
            L,
            self.L_col_labels
        )
        self.primal.create_primal_problem()
        self.primal.model.update()
        self.primal.model.optimize()

        return self





