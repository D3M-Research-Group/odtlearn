from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# Include Necessary imports in the same folder
from trees.utils.Tree import Tree
from trees.utils.RobustOCT import RobustOCT
from trees.utils.RobustTreeUtils import mycallback, check_integer, check_same_as_X
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

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, depth=1, time_limit=1800):
        self.depth = depth
        self.time_limit = time_limit

    def extract_metadata(self, X, y):
        """ A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API

        """
        if isinstance(X, pd.DataFrame):
            self.X_col_labels = X.columns
            self.X = X
        else:
            self.X_col_labels = np.arange(0, self.X.shape[1])
            self.X = pd.DataFrame(X, columns=self.X_col_labels)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.y = y.values
        else:
            self.y = y        

    def fit(self, X, y, costs=None, budget=0):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        costs: array-like, shape (n_samples, n_features)
            The costs of uncertainty 
        budget: float
            The budget of uncertainty

        Returns
        -------
        self : object
            Returns self.
        """
        self.extract_metadata(X, y)
        X, y = check_X_y(X, y)
        check_integer(self.X)

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Instantiate tree object here
        tree = Tree(self.depth)

        # Set default for costs of uncertainty if needed
        if costs is not None:
            self.costs = check_same_as_X(self.X, self.X_col_labels, costs, "Uncertainty costs")
        else:
            # By default, set costs to be budget + 1 (i.e. no uncertainty)
            gammas_df = deepcopy(self.X).astype('float')
            for col in gammas_df.columns:
                gammas_df[col].values[:] = budget+1
            self.costs = gammas_df

        # Budget of uncertainty
        if budget < 0:
            raise ValueError("Budget of uncertainty must be nonnegative")
        self.budget = budget

        # Code for setting up and running the MIP goes here.
        # Note that we are taking X and y as array-like objects
        self.start_time = time.time()
        master = RobustOCT(self.X, self.y, tree, self.X_col_labels, self.classes_, 
            self.time_limit, self.costs, self.budget)
        master.create_master_problem()
        master.model.update()
        master.model.optimize(mycallback)
        self.end_time = time.time()
        self.solving_time = self.end_time - self.start_time

        # Store fitted Gurobi model
        self.model = master

        # `fit` should always return `self`
        return self

    def get_prediction(self, X):
        b = self.model.model.getAttr("X", self.model.b)
        w = self.model.model.getAttr("X", self.model.w)
        prediction = []
        for i in X.index:
            # Get prediction value
            node = 1
            while True:
                terminal = False
                for k in self.model.labels:
                    if w[node, k] > 0.5: # w[n,k] == 1
                        prediction += [k]
                        terminal = True
                        break
                if terminal:
                    break
                else:
                    for (f, theta) in self.model.f_theta_indices:
                        if b[node, f, theta] > 0.5: # b[n,f]== 1
                            if X.at[i, f] >= theta + 1:
                                node = self.model.tree.get_right_children(node)
                            else:
                                node = self.model.tree.get_left_children(node)
                            break
        return np.array(prediction)

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
        check_is_fitted(self, ['model'])

        # Convert to dataframe
        df_test = check_same_as_X(self.X, self.X_col_labels, X, "Test covariates")
        check_integer(df_test)

        return self.get_prediction(df_test)
