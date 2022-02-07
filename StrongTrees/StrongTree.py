import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import time
from StrongTrees.StrongTreeUtils import check_columns_match
# Include Tree.py and FlowOCT.py in StrongTrees folder
from Tree import Tree
from FlowOCT import FlowOCT
from StrongTreeUtils import get_predicted_value, check_binary


class StrongTreeClassifier(ClassifierMixin, BaseEstimator):
    """ 

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

    def __init__(self, depth, time_limit, _lambda, num_threads=1):
        # this is where we will initialize the values we want users to provide
        self.depth = depth
        self.time_limit = time_limit,
        self._lambda = _lambda
        self.num_threads = num_threads
        self.mode = "classification"
        self.X_col_labels = None
        self.X_col_dtypes = None
        self.y_dtypes = None

    def extract_metadata(self, X, y):
        """ A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API

        """
        if isinstance(X, pd.Dataframe):
            self.X_col_labels = X.columns
            self.X_col_dtypes = X.dtypes
            self.X = X
        else:
            self.X_col_labels = np.arange(0, self.X.shape[1])
            self.X = pd.Dataframe(X, columns=self.X_col_labels)

        if isinstance(self.y, [pd.Series, pd.DataFrame]):
            self.y = y.values
            self.y_dtypes = y.dtypes
            self.labels = np.unique(self.y)
        else:
            self.y = y
            self.labels = np.unique(self.y)

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

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

        # keep original data, we only modify the data in FlowOCT
        self.X_ = X
        self.y_ = y

        # Instantiate tree object here
        tree = Tree(self.depth)

        # Code for setting up and running the MIP goes here.
        # Note that we are taking X and y as array-like objects
        self.start_time = time.time()
        self.primal = FlowOCT(X, y, tree, self._lambda,
                              self.time_limit, self.mode, self.num_threads)
        self.primal.create_primal_problem()
        self.primal.model.update()
        self.primal.model.optimize()
        self.end_time = time.time()
        # solving_time or other potential parameters of interest can be stored
        # within the class: self.solving_time
        self.solving_time = self.end_time - self.start_time

        # Here we will want to store these values and any other variables
        # needed for making predictions later
        self.b_value = self.primal.model.getAttr("X", self.primal.b)
        self.beta_value = self.primal.model.getAttr("X", self.primal.beta)
        self.p_value = self.primal.model.getAttr("X", self.primal.p)
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

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
        check_is_fitted(self, ['X_', 'y_'])

        check_columns_match(self.X_col_labels, X)
        self.X_predict_col_names = X.columns

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        prediction = get_predicted_value(
            self.model, X, self.b, self.beta_value, self.p)
        return prediction
