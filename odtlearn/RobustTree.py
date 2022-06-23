from copy import deepcopy
import pandas as pd
import time

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from odtlearn.utils.Tree import Tree
from odtlearn.utils.robusttree_formulation import RobustOCT

from odtlearn.utils.RobustTreeUtils import (
    robust_tree_callback,
    check_integer,
    check_same_as_X,
)
from odtlearn.tree_classifier import TreeClassifier


class RobustTreeClassifier(TreeClassifier):
    """An optimal robust decision tree classifier, fitted on a given integer-valued
    data set and a given cost-and-budget uncertainty set to produce a tree robust
    against distribution shifts.

    Parameters
    ----------
    depth : int, default=1
        A parameter specifying the depth of the tree
    time_limit : int, default=1800
        The given time limit for solving the MIP in seconds
    num_threads: int, default=None
        The number of threads the solver should use. If not specified,
        solver uses Gurobi's default number of threads

    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : array-like, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    costs : pandas.DataFrame, shape (n_samples, n_features)
        The uncertainty costs used during fitting
    budget : float
        The uncertainty budget used during fitting
    b_value : nddict, shape (tree_internal_nodes, X_features, X_features_cutoffs)
        The values of decision variable b -- the branching decisions of the tree
    w_value : nddict, shape (tree_nodes)
        The values of decision variable w -- the predictions at the tree's nodes
    model : gurobipy.Model
        The trained Gurobi model, with solver information and
        decision variable information (`b` for branching variables,
        `w` for assignment variables)
    """

    def __init__(self, depth=1, time_limit=1800, num_threads=None) -> None:
        super().__init__(depth, time_limit, num_threads)

    def fit(self, X, y, costs=None, budget=-1, verbose=True):
        """Fit an optimal robust classification tree given data, labels,
        costs of uncertainty, and budget of uncertainty

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        costs : array-like, shape (n_samples, n_features), default = budget + 1
            The costs of uncertainty
        budget : float, default = -1
            The budget of uncertainty
        verbose : bool, default = True
            Flag for logging Gurobi outputs

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
            self.costs = check_same_as_X(
                self.X, self.X_col_labels, costs, "uncertainty costs"
            )
            self.costs.set_index(pd.Index(range(costs.shape[0])), inplace=True)
            # Also check if indices are the same
            if self.X.shape[0] != self.costs.shape[0]:
                raise ValueError(
                    (
                        f"Input covariates has {self.X.shape[0]} samples, "
                        f"but uncertainty costs has {self.costs.shape[0]}"
                    )
                )
        else:
            # By default, set costs to be budget + 1 (i.e. no uncertainty)
            gammas_df = deepcopy(self.X).astype("float")
            for col in gammas_df.columns:
                gammas_df[col].values[:] = budget + 1
            self.costs = gammas_df

        # Budget of uncertainty
        self.budget = budget

        # Code for setting up and running the MIP goes here.
        # Note that we are taking X and y as array-like objects
        self.start_time = time.time()
        self.grb_model = RobustOCT(
            self.X,
            self.y,
            tree,
            self.X_col_labels,
            self.classes_,
            self.costs,
            self.budget,
            self.time_limit,
            self.num_threads,
            verbose,
        )
        self.grb_model.create_main_problem()
        self.grb_model.model.update()
        self.grb_model.model.optimize(robust_tree_callback)
        self.end_time = time.time()
        self.solving_time = self.end_time - self.start_time

        # Store fitted Gurobi model
        self.b_value = self.grb_model.model.getAttr("X", self.grb_model.b)
        self.w_value = self.grb_model.model.getAttr("X", self.grb_model.w)

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """Given the input covariates, predict the class labels of each sample
        based on the fitted optimal robust classification tree

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
        check_is_fitted(self, ["grb_model"])

        # Convert to dataframe
        df_test = check_same_as_X(self.X, self.X_col_labels, X, "test covariates")
        check_integer(df_test)

        return self._get_prediction(df_test)
