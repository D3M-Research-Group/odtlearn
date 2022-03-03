import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (check_X_y,
                                      check_array,
                                      _assert_all_finite,
                                      check_is_fitted,
                                      column_or_1d,
                                      check_consistent_length)
from sklearn.utils.multiclass import unique_labels
import time
from trees.utils.StrongTreeUtils import (
    check_columns_match,
    check_binary,
    benders_callback,
)

# Include Tree.py, FlowOCT.py and BendersOCT.py in StrongTrees folder
from trees.utils.Tree import Tree
from trees.utils.StrongTreeFlowOCT import FlowOCT
from trees.utils.StrongTreeBendersOCT import BendersOCT
from trees.utils.PrescriptiveTreesMIP import FlowOPT_IPW, FlowOPT_Robust


class PrescriptiveTreeClassifier(ClassifierMixin, BaseEstimator):
    """
    Parameters
    ----------
    depth : intClassifierMixin
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
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, depth, time_limit, method='IPW', num_threads=None):
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

        :param X: The input/training data
        :param t: A vector or array-like object for the treatment assignments

        """
        if isinstance(X, pd.DataFrame):
            self.X_col_labels = X.columns
            self.X_col_dtypes = X.dtypes
        else:
            self.X_col_labels = np.arange(0, X.shape[1])

        self.treatments = np.unique(t)


    def get_node_status(self, labels, column_names, b, beta, p, n):
        """
        This function give the status of a given node in a tree. By status we mean whether the node
            1- is pruned? i.e., we have made a prediction at one of its ancestors
            2- is a branching node? If yes, what feature do we branch on
            3- is a leaf? If yes, what is the prediction at this node?
        :param labels: the unique values of the response variable y
        :param column_names: the column names of the data set X
        :param b: The values of branching decision variable b
        :param beta: The values of prediction decision variable beta
        :param p: The values of decision variable p
        :param n: A valid node index in the tree
        :return: pruned, branching, selected_feature, leaf, value
        pruned=1 iff the node is pruned
        branching = 1 iff the node branches at some feature f
        selected_feature: The feature that the node branch on
        leaf = 1 iff node n is a leaf in the tree
        value: if node n is a leaf, value represent the prediction at this node
        """

        pruned = False
        branching = False
        leaf = False
        value = None
        selected_feature = None

        p_sum = 0
        for m in self.tree.get_ancestors(n):
            p_sum = p_sum + p[m]
        if p[n] > 0.5:  # leaf
            leaf = True
            for k in labels:
                if beta[n, k] > 0.5:
                    value = k
        elif p_sum == 1:  # Pruned
            pruned = True

        if n in self.tree.Nodes:
            if (pruned is False) and (leaf is False):  # branching
                for f in column_names:
                    if b[n, f] > 0.5:
                        selected_feature = f
                        branching = True

        return pruned, branching, selected_feature, leaf, value

    def get_predicted_value(self, X, b, beta, p):
        """
        This function returns the predicted value for a given datapoint
        :param X: The dataset we want to compute accuracy for
        :param b: The value of decision variable b
        :param beta: The value of decision variable beta
        :param p: The value of decision variable p
        :return: The predicted value for all datapoints in dataset X
        """
        predicted_values = []
        for i in range(X.shape[0]):
            current = 1
            while True:
                pruned, branching, selected_feature, leaf, value = self.get_node_status(
                    self.treatments, self.X_col_labels, b, beta, p, current
                )
                if leaf:
                    predicted_values.append(value)
                    break
                elif branching:
                    selected_feature_idx = np.where(
                        self.X_col_labels == selected_feature
                    )
                    # Raise assertion error we don't have a column that matches
                    # the selected feature or more than one column that matches
                    assert (
                        len(selected_feature_idx) == 1
                    ), f"Found {len(selected_feature_idx)} columns matching the selected feature {selected_feature}"
                    if X[i, selected_feature_idx] == 1:  # going right on the branch
                        current = self.tree.get_right_children(current)
                    else:  # going left on the branch
                        current = self.tree.get_left_children(current)
        return predicted_values

    def check_helpers(self, X, ipw, y_hat):
        """
        This function checks the propensity weights and counterfactual predictions
        :param X: The input/training data
        :param ipw: A vector or array-like object for inverse propensity weights. Only needed when running IPW/DR
        :param y_hat: A multi-dimensional array-like object for counterfactual predictions. Only needed when running DM/DR
        :return: The converted versions of ipw and y_hat after passing the series of checks
        """
        if self.method in ['IPW', 'DR']:
            assert (
                    ipw is not None
            ), f"Inverse propensity weights cannot be None"

            ipw = column_or_1d(ipw, warn=True)

            if ipw.dtype.kind == 'O':
                ipw = ipw.astype(np.float64)

            assert (
                min(ipw) > 0 and max(ipw) <= 1
            ), f"Inverse propensity weights must be in the range (0, 1]"

            check_consistent_length(X, ipw)

        if self.method in ['DM', 'DR']:
            assert (
                    y_hat is not None
            ), f"Counterfactual estimates cannot be None"

            y_hat = check_array(y_hat)

            # y_hat has to have as many columns as there are treatments
            assert(
                    y_hat.shape[1] == len(self.treatments)
            ), f"Found counterfactual estimates for {y_hat.shape[1]} treatments. There are {len(self.treatments)} unique treatments in the data"

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
        if y.dtype.kind == 'O':
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
        except Exception as e:
            print('The set of treatments must be discrete.')

        assert (
                min(t) == 0 and max(t) == len(set(t)) - 1
        ), 'The set of treatments must be discrete starting from {0, 1, ...}'

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

        if self.method == 'IPW':
            self.primal = FlowOPT_IPW(
                X,
                t,
                y,
                ipw,
                self.treatments,
                self.tree,
                self.X_col_labels,
                self.time_limit,
                self.num_threads
            )

        elif self.method == 'DM':
            self.primal = FlowOPT_Robust(
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
                self.num_threads
            )

        elif self.method == 'DR':
            self.primal = FlowOPT_Robust(
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
                self.num_threads
            )

        self.primal.create_main_problem()
        self.primal.model.update()
        self.primal.model.optimize()


        self.end_time = time.time()
        # solving_time or other potential parameters of interest can be stored
        # within the class: self.solving_time
        self.solving_time = self.end_time - self.start_time

        # Here we will want to store these values and any other variables
        # needed for making predictions later
        self.b_value = self.primal.model.getAttr("X", self.primal.b)
        self.w_value = self.primal.model.getAttr("X", self.primal.w)
        self.p_value = self.primal.model.getAttr("X", self.primal.p)

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

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self.X_col_labels, X)

        prediction = self.get_predicted_value(
            X,
            self.b_value,
            self.w_value,
            self.p_value,
        )
        return prediction
