import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# Include Necessary imports in the same folder
import pandas as pd
from Tree import Tree
from RobustOCT import RobustOCT
from RobustTreeUtils import *
import time

class RobustTreeClassifier(ClassifierMixin, BaseEstimator):
    """ An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    depth : int, default=1
        A parameter specifying the depth of the tree
    time_limit : int, default=1800
        The given time limit for solving the MIP in seconds
    q : float, default=1.0
        Mean of probability of feature certainty
    s : float, default=0.0
        Standard deviation of probability of feature certainty
    p : float, default=1.0
        Probability of label certainty
    l : float, default=1.0
        Probability testing threshold
    seed : int
        Seed for uncertainty set parameter generation

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, depth=1, time_limit=1800, q=1.0, s=0.0, p=1.0, l=1.0, seed=None):
        self.depth = depth
        self.time_limit = time_limit
        self.q = q
        self.s = s
        self.p = p
        self.l = l
        self.seed = seed

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
        self.extract_metadata(X, y)
        X, y = check_X_y(X, y, accept_sparse=True)

        # Check Integer
        
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y


        # Instantiate tree object here
        tree = Tree(self.depth)

        # Code for setting up and running the MIP goes here.
        # Note that we are taking X and y as array-like objects
        self.start_time = time.time()
        master = RobustOCT(X, y, tree, self.time_limit, 
            self.q, self.s, self.p, self.l, self.seed)
        master.create_master_problem()
        master.model.update()
        master.model.optimize(mycallback)
        self.end_time = time.time()
        self.solving_time = self.end_time - self.start_time

        # Store fitted Gurobi model
        self.model = master

        # `fit` should always return `self`
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
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        # Convert to dataframe
        data=X
        if not isinstance(X, pd.Dataframe):
            data = pd.Dataframe(X, columns=np.arange(0, self.X.shape[1]))

        return get_prediction(self.model, data)
