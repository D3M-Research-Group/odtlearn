import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    _assert_all_finite,
    check_is_fitted,
    column_or_1d,
    check_consistent_length,
)
import time
from odtlearn.utils.StrongTreeUtils import (
    check_columns_match,
    check_binary,
)
from odtlearn.utils.PrescriptiveTreeUtils import get_predicted_value
from odtlearn.utils.Tree import Tree
from odtlearn.utils.PrescriptiveTreesMIP import FlowOPT_IPW, FlowOPT_Robust


class PrescriptiveTreeClassifier(ClassifierMixin, BaseEstimator):
    """An optimal decision tree that prescribes treatments (as opposed to predicting class labels),
    fitted on a binary-valued observational data set. 

    Parameters
    ----------
    depth : int
        A parameter specifying the depth of the tree
    time_limit : int
        The given time limit for solving the MIP in seconds
    method : str, default='IPW'
        The method of Prescriptive Trees to run. Choices in ('IPW', 'DM', 'DR), which represents the
        inverse propensity weighting, direct method, and doubly robust methods, respectively
    num_threads: int, default=None
        The number of threads the solver should use

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The observed outcomes passed during :meth:`fit`.
    t_ : ndarray, shape (n_samples,)
        The treatments passed during :meth: `fit`.
    treatments : set, shape (n_treatments,)
        The set of unique treatments seen at :meth:`fit`.
    solving_time : float
        The amount of time the mixed-integer program took to solve
    b_value : nddict, shape (tree_internal_nodes, X_features)
        The values of decision variable b -- the branching decisions of the tree
    w_value : nddict, shape (tree_nodes, treatments_set)
        The values of decision variable w -- the treatment decisions at the tree's nodes
    p_value : nddict, shape (tree_nodes,)
        The values of decision variable p -- whether or not a tree's nodes branch or assign treatment
    """

    def __init__(self, depth, time_limit, method="IPW", num_threads=None):
        # this is where we will initialize the values we want users to provide
        self.depth = depth
        self.time_limit = time_limit
        self.num_threads = num_threads
        self.method = method

        self.X_col_labels = None
        self.X_col_dtypes = None
        self.y_dtypes = None

        self.treatments = None
        self.ipw = None
        self.y_hat = None

    def extract_metadata(self, X, t):
        """A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API

        Parameters
        ----------
        X :
            The input/training data
        t :
            A vector or array-like object for the treatment assignments

        """
        if isinstance(X, pd.DataFrame):
            self.X_col_labels = X.columns
            self.X_col_dtypes = X.dtypes
        else:
            self.X_col_labels = np.arange(0, X.shape[1])

        self.treatments = np.unique(t)

    def check_helpers(self, X, ipw, y_hat):
        """
        This function checks the propensity weights and counterfactual predictions

        Parameters
        ----------
        X: The input/training data
        ipw: A vector or array-like object for inverse propensity weights. Only needed when running IPW/DR
        y_hat: A multi-dimensional array-like object for counterfactual predictions. Only needed when running DM/DR

        :return: The converted versions of ipw and y_hat after passing the series of checks
        """
        if self.method in ["IPW", "DR"]:
            assert ipw is not None, "Inverse propensity weights cannot be None"

            ipw = column_or_1d(ipw, warn=True)

            if ipw.dtype.kind == "O":
                ipw = ipw.astype(np.float64)

            assert (
                min(ipw) > 0 and max(ipw) <= 1
            ), "Inverse propensity weights must be in the range (0, 1]"

            check_consistent_length(X, ipw)

        if self.method in ["DM", "DR"]:
            assert y_hat is not None, "Counterfactual estimates cannot be None"

            y_hat = check_array(y_hat)

            # y_hat has to have as many columns as there are treatments

            assert y_hat.shape[1] == len(
                self.treatments
            ), f"Found counterfactual estimates for {y_hat.shape[1]} treatments. \
            There are {len(self.treatments)} unique treatments in the data"

            check_consistent_length(X, y_hat)

        return ipw, y_hat

    def check_y(self, X, y):
        """
        This function checks the shape and contents of the observed outcomes
        :param X: The input/training data
        :param y: A vector or array-like object for the observed outcomes corresponding to treatment t
        :return: The converted version of y after passing the series of checks
        """
        # check consistent length
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
        if y.dtype.kind == "O":
            y = y.astype(np.float64)

        check_consistent_length(X, y)
        return y

    def fit(self, X, t, y, ipw=None, y_hat=None):
        """Method to fit the PrescriptiveTree class on the data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        t : array-like, shape (n_samples,)
            The treatment values. An array of int.
        y : array-like, shape (n_samples,)
            The observed outcomes upon given treatment t. An array of int.
        ipw : array-like, shape (n_samples,)
            The inverse propensity weight estimates. An array of floats in [0, 1].
        y_hat: array-like, shape (n_samples, n_treatments)
            The counterfactual predictions.

        Returns
        -------
        self : object
            Returns self.
        """
        # check if self.method is one of the three options

        # store column information and dtypes if any
        self.extract_metadata(X, t)

        # this function returns converted X and t but we retain metadata
        X, t = check_X_y(X, t)

        # need to check that t is discrete, and/or convert -- starts from 0 in accordance with indexing rule
        try:
            t = t.astype(int)
        except Exception:
            print("The set of treatments must be discrete.")

        assert (
            min(t) == 0 and max(t) == len(set(t)) - 1
        ), "The set of treatments must be discrete starting from {0, 1, ...}"

        # we also need to check on y and ipw/y_hat depending on the method chosen
        y = self.check_y(X, y)
        ipw, y_hat = self.check_helpers(X, ipw, y_hat)

        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)

        # Store the classes seen during fit
        self.treatments_ = np.unique(t)

        # keep original data
        self.X_ = X
        self.y_ = y
        self.t_ = t

        # Instantiate tree object here
        self.tree = Tree(self.depth)

        # Code for setting up and running the MIP goes here.
        # Note that we are taking X and y as array-like objects
        self.start_time = time.time()

        if self.method == "IPW":
            self.grb_model = FlowOPT_IPW(
                X,
                t,
                y,
                ipw,
                self.treatments,
                self.tree,
                self.X_col_labels,
                self.time_limit,
                self.num_threads,
            )

        elif self.method == "DM":
            self.grb_model = FlowOPT_Robust(
                X,
                t,
                y,
                ipw,
                y_hat,
                False,
                self.treatments,
                self.tree,
                self.X_col_labels,
                self.time_limit,
                self.num_threads,
            )

        elif self.method == "DR":
            self.grb_model = FlowOPT_Robust(
                X,
                t,
                y,
                ipw,
                y_hat,
                True,
                self.treatments,
                self.tree,
                self.X_col_labels,
                self.time_limit,
                self.num_threads,
            )

        self.grb_model.create_main_problem()
        self.grb_model.model.update()
        self.grb_model.model.optimize()

        self.end_time = time.time()
        # solving_time or other potential parameters of interest can be stored
        # within the class: self.solving_time
        self.solving_time = self.end_time - self.start_time

        # Here we will want to store these values and any other variables
        # needed for making predictions later
        self.b_value = self.grb_model.model.getAttr("X", self.grb_model.b)
        self.w_value = self.grb_model.model.getAttr("X", self.grb_model.w)
        self.p_value = self.grb_model.model.getAttr("X", self.grb_model.p)

        # Return the classifier
        return self

    def predict(self, X):
        """Method for making predictions using a PrescriptiveTree classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        t : ndarray, shape (n_samples,)
            The predicted treatment for the input samples.
        """
        # Check if fit had been called
        check_is_fitted(self, ["X_", "y_", "t_"])

        if isinstance(X, pd.DataFrame):
            self.X_predict_col_names = X.columns
        else:
            self.X_predict_col_names = np.arange(0, X.shape[1])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self.X_col_labels, X)

        prediction = get_predicted_value(
            self.grb_model,
            X,
            self.b_value,
            self.w_value,
            self.p_value,
        )
        return prediction
