import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from odtlearn.utils.StrongTreeUtils import (
    check_columns_match,
    check_binary,
    benders_callback,
    get_predicted_value,
    print_tree_util,
)
from odtlearn.utils.Tree import Tree
from odtlearn.utils.strongtree_formulation import FlowOCT, BendersOCT
from odtlearn.utils.TreePlotter import MPLPlotter
from odtlearn.tree_classifier import TreeClassifier


class StrongTreeClassifier(ClassifierMixin, BaseEstimator):
    """A strong optimal classification tree fitted on a given binary-valued
    data set.

    Parameters
    ----------
    depth : int, default=1
        A parameter specifying the depth of the tree
    time_limit : int, default=60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default= 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    benders_oct: bool, default=True
        Use benders problem formulation.
    obj_mode: str, default="acc"
        Set objective priority. If "acc", maximize the accuracy, if "balance" maximize the balanced accuracy
    num_threads: int, default=None
        The number of threads the solver should use. If no argument is supplied, Gurobi will use all available threads.


    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    b_value : a dictionary containing the value of the decision variables b,
    where b_value[(n,f)] is the value of b at node n and feature f
    w_value : a dictionary containing the value of the decision variables w,
    where w_value[(n,k)] is the value of w at node n and class label k
    p_value : a dictionary containing the value of the decision variables p,
    where p_value[n] is the value of p at node n
    grb_model : gurobipy.Model
        The fitted Gurobi model

    Examples
    --------
    >>> from odtlearn.StrongTree import StrongTreeClassifier
    >>> import numpy as np
    >>> X = np.arange(200).reshape(100, 2)
    >>> y = np.random.randint(2, size=100)
    >>> stcl = StrongTreeClassifier(depth = 1, _lambda = 0, time_limit = 60, num_threads = 1)
    >>> stcl.fit(X, y)

    """

    def __init__(
        self,
        depth=1,
        time_limit=60,
        _lambda=0,
        benders_oct=True,
        obj_mode="acc",
        num_threads=None,
    ):
        # this is where we will initialize the values we want users to provide
        self.depth = depth
        self.time_limit = time_limit
        self._lambda = _lambda
        self.num_threads = num_threads
        self.benders_oct = benders_oct
        self.obj_mode = obj_mode

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
            self.X_col_labels = np.arange(0, X.shape[1])

        self.labels = np.unique(y)

    def fit(self, X, y, verbose=False):
        """Fit a StrongTree using the supplied data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        verbose : bool, default = True
            Flag for logging Gurobi outputs

        Returns
        -------
        self : object
            Returns self.
        """
        # store column information and dtypes if any
        self.extract_metadata(X, y)

        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)

        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # keep original data
        self.X_ = X
        self.y_ = y

        # Instantiate tree object here
        tree = Tree(self.depth)

        # Code for setting up and running the MIP goes here.
        # Note that we are taking X and y as array-like objects

        if self.benders_oct:
            self.grb_model = BendersOCT(
                X,
                y,
                tree,
                self.X_col_labels,
                self.labels,
                self._lambda,
                self.obj_mode,
                self.time_limit,
                self.num_threads,
                verbose,
            )
            self.grb_model.create_main_problem()
            self.grb_model.model.update()
            self.grb_model.model.optimize(benders_callback)
        else:
            self.grb_model = FlowOCT(
                X,
                y,
                tree,
                self.X_col_labels,
                self.labels,
                self._lambda,
                self.obj_mode,
                self.time_limit,
                self.num_threads,
                verbose,
            )
            self.grb_model.create_primal_problem()
            self.grb_model.model.update()
            self.grb_model.model.optimize()

        # solving_time or other potential parameters of interest can be stored
        # within the class: self.solving_time
        self.solving_time = self.grb_model.model.getAttr("Runtime")

        # Here we will want to store these values and any other variables
        # needed for making predictions later
        self.b_value = self.grb_model.model.getAttr("X", self.grb_model.b)
        self.w_value = self.grb_model.model.getAttr("X", self.grb_model.w)
        self.p_value = self.grb_model.model.getAttr("X", self.grb_model.p)
        # Return the classifier
        return self

    def predict(self, X):
        """Classify test points using the StrongTree classifier

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

    def print_tree(self):
        """
        This function print the derived tree with the branching features and the predictions asserted for each node


        Returns
        -------
        Print out the tree in the console
        """

        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])
        print_tree_util(self.grb_model, self.b_value, self.w_value, self.p_value)

    def plot_tree(
        self,
        label="all",
        filled=True,
        rounded=False,
        precision=3,
        ax=None,
        fontsize=None,
        color_dict={"node": None, "leaves": []},
        edge_annotation=True,
        arrow_annotation_font_scale=0.5,
        debug=False,
    ):
        check_is_fitted(self, ["X_", "y_"])
        exporter = MPLPlotter(
            self.grb_model,
            self.X_col_labels,
            self.b_value,
            self.w_value,
            self.p_value,
            self.grb_model.tree.depth,
            self.classes_,
            label=label,
            filled=filled,
            rounded=rounded,
            precision=precision,
            fontsize=fontsize,
            color_dict=color_dict,
            edge_annotation=edge_annotation,
            arrow_annotation_font_scale=arrow_annotation_font_scale,
            debug=debug,
        )
        return exporter.export(ax=ax)


class _StrongTreeClassifier(TreeClassifier):
    """A strong optimal classification tree fitted on a given binary-valued
    data set.

    Parameters
    ----------
    depth : int, default=1
        A parameter specifying the depth of the tree
    time_limit : int, default=60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default= 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    benders_oct: bool, default=True
        Use benders problem formulation.
    obj_mode: str, default="acc"
        Set objective priority. If "acc", maximize the accuracy, if "balance" maximize the balanced accuracy
    num_threads: int, default=None
        The number of threads the solver should use. If no argument is supplied, Gurobi will use all available threads.


    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    b_value : a dictionary containing the value of the decision variables b,
    where b_value[(n,f)] is the value of b at node n and feature f
    w_value : a dictionary containing the value of the decision variables w,
    where w_value[(n,k)] is the value of w at node n and class label k
    p_value : a dictionary containing the value of the decision variables p,
    where p_value[n] is the value of p at node n
    grb_model : gurobipy.Model
        The fitted Gurobi model

    Examples
    --------
    >>> from odtlearn.StrongTree import StrongTreeClassifier
    >>> import numpy as np
    >>> X = np.arange(200).reshape(100, 2)
    >>> y = np.random.randint(2, size=100)
    >>> stcl = StrongTreeClassifier(depth = 1, _lambda = 0, time_limit = 60, num_threads = 1)
    >>> stcl.fit(X, y)

    """

    def __init__(
        self,
        depth=1,
        time_limit=60,
        _lambda=0,
        benders_oct=True,
        obj_mode="acc",
        num_threads=None,
    ) -> None:
        super().__init__(depth, time_limit, num_threads)
        self.benders_oct = benders_oct
        self.obj_mode = obj_mode
        self._lambda = _lambda

    def fit(self, X, y, verbose=False):
        """Fit a StrongTree using the supplied data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        verbose : bool, default = True
            Flag for logging Gurobi outputs

        Returns
        -------
        self : object
            Returns self.
        """
        # store column information and dtypes if any
        self.extract_metadata(X, y)

        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)

        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # keep original data
        self.X_ = X
        self.y_ = y

        # Instantiate tree object here
        tree = Tree(self.depth)

        # Code for setting up and running the MIP goes here.
        # Note that we are taking X and y as array-like objects

        if self.benders_oct:
            self.grb_model = BendersOCT(
                X,
                y,
                tree,
                self.X_col_labels,
                self.labels,
                self._lambda,
                self.obj_mode,
                self.time_limit,
                self.num_threads,
                verbose,
            )
            self.grb_model.create_main_problem()
            self.grb_model.model.update()
            self.grb_model.model.optimize(benders_callback)
        else:
            self.grb_model = FlowOCT(
                X,
                y,
                tree,
                self.X_col_labels,
                self.labels,
                self._lambda,
                self.obj_mode,
                self.time_limit,
                self.num_threads,
                verbose,
            )
            self.grb_model.create_primal_problem()
            self.grb_model.model.update()
            self.grb_model.model.optimize()

        # solving_time or other potential parameters of interest can be stored
        # within the class: self.solving_time
        self.solving_time = self.grb_model.model.getAttr("Runtime")

        # Here we will want to store these values and any other variables
        # needed for making predictions later
        self.b_value = self.grb_model.model.getAttr("X", self.grb_model.b)
        self.w_value = self.grb_model.model.getAttr("X", self.grb_model.w)
        self.p_value = self.grb_model.model.getAttr("X", self.grb_model.p)
        # Return the classifier
        return self

    def predict(self, X):
        """Classify test points using the StrongTree classifier

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
        check_is_fitted(self, ["grb_model"])

        if isinstance(X, pd.DataFrame):
            self.X_predict_col_names = X.columns
        else:
            self.X_predict_col_names = np.arange(0, X.shape[1])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self.X_col_labels, X)

        # private function from TreeClassifier
        prediction = self._get_prediction(X)

        return prediction
