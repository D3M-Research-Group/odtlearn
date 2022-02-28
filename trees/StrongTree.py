import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
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


class StrongTreeClassifier(ClassifierMixin, BaseEstimator):
    """TO-DO: NEED DESCRIPTION OF THE CLASS HERE.

    Parameters
    ----------
    depth : int, default=1
        A parameter specifying the depth of the tree
    time_limit : int
        The given time limit for solving the MIP in seconds
    _lambda : int
        The regularization parameter in the objective
    num_threads: int, default=1
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

    def __init__(self, depth, time_limit, _lambda, benders_oct=False, num_threads=None):
        # this is where we will initialize the values we want users to provide
        self.depth = depth
        self.time_limit = time_limit
        self._lambda = _lambda
        self.num_threads = num_threads
        self.benders_oct = benders_oct

        self.X_col_labels = None
        self.X_col_dtypes = None
        self.y_dtypes = None

    def extract_metadata(self, X, y):
        """A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API

        """
        if isinstance(X, pd.DataFrame):
            self.X_col_labels = X.columns
            self.X_col_dtypes = X.dtypes
        else:
            self.X_col_labels = np.arange(0, self.X.shape[1])

        self.labels = np.unique(y)

    def get_node_status(self, labels, column_names, b, w, p, n):
        """
        This function give the status of a given node in a tree. By status we mean whether the node
            1- is pruned? i.e., we have made a prediction at one of its ancestors
            2- is a branching node? If yes, what feature do we branch on
            3- is a leaf? If yes, what is the prediction at this node?
        :param labels: the unique values of the response variable y
        :param column_names: the column names of the data set X
        :param b: The values of branching decision variable b
        :param w: The values of prediction decision variable w
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
                if w[n, k] > 0.5:
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

    def get_predicted_value(self, X, b, w, p):
        """
        This function returns the predicted value for a given dataset
        :param X: The dataset we want to compute accuracy for
        :param b: The value of decision variable b
        :param w: The value of decision variable w
        :param p: The value of decision variable p
        :return: The predicted value for all datapoints in dataset X
        """
        predicted_values = []
        for i in range(X.shape[0]):
            current = 1
            while True:
                pruned, branching, selected_feature, leaf, value = self.get_node_status(
                    self.labels, self.X_predict_col_names, b, w, p, current
                )
                if leaf:
                    predicted_values.append(value)
                    break
                elif branching:
                    selected_feature_idx = np.where(
                        self.X_predict_col_names == selected_feature
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
        return np.array(predicted_values)

    def fit(self, X, y):
        """TO-DO: NEED DESCRIPTION OF METHOD HERE.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # store column information and dtypes if any
        self.extract_metadata(X, y)
        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)
        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # keep original data
        self.X_ = X
        self.y_ = y

        # Instantiate tree object here
        self.tree = Tree(self.depth)

        # Code for setting up and running the MIP goes here.
        # Note that we are taking X and y as array-like objects
        self.start_time = time.time()
        if self.benders_oct:
            self.grb_model = BendersOCT(
                X,
                y,
                self.tree,
                self.X_col_labels,
                self.labels,
                self._lambda,
                self.time_limit,
                self.num_threads,
            )
            self.grb_model.create_main_problem()
            self.grb_model.model.update()
            self.grb_model.model.optimize(benders_callback)
        else:
            self.grb_model = FlowOCT(
                X,
                y,
                self.tree,
                self.X_col_labels,
                self.labels,
                self._lambda,
                self.time_limit,
                self.num_threads,
            )
            self.grb_model.create_primal_problem()
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
        """TO-DO: NEED DESCRIPTION OF METHOD HERE.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        self.X_predict_col_names = X.columns
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
