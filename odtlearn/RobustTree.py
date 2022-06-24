from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from odtlearn.utils.TreePlotter import MPLPlotter
from odtlearn.utils.Tree import Tree
from odtlearn.utils.RobustOCT import RobustOCT
from odtlearn.utils.RobustTreeUtils import mycallback, check_integer, check_same_as_X
import time


class RobustTreeClassifier(ClassifierMixin, BaseEstimator):
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

    def __init__(self, depth=1, time_limit=1800, num_threads=None):
        self.depth = depth
        self.time_limit = time_limit
        self.num_threads = num_threads

    def extract_metadata(self, X, y):
        """A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API
        """
        if isinstance(X, pd.DataFrame):
            self.X_col_labels = X.columns
            self.X = X
        else:
            self.X_col_labels = np.arange(0, X.shape[1])
            self.X = pd.DataFrame(X, columns=self.X_col_labels)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.y = y.values
        else:
            self.y = y

        # Strip indices in training data into integers
        self.X.set_index(pd.Index(range(X.shape[0])), inplace=True)

    def probabilities_to_cost(self, prob, threshold=1):
        """Convert probabilities of certainty and levels of robustness
        to costs and budget values for fitting a robust tree

        Parameters
        ----------
        prob : array-like, shape (n_samples, n_features)
            A 2D matrix of probabilities where the value of row i and column f
            is the probability that the value for sample i and feature f is 
            certain. Each entry must be between 0 and 1 (inclusive).
        threshold : float, default = 1
            The threshold that tunes the level of robustness, between 0 (exclusive,
            complete robustness) and 1 (inclusive, no robustness).
        Returns
        -------
        costs : pandas DataFrame
            The costs of uncertainty to use for training
        budget: float
            The budget of uncertainty based on the given threshold information
            and number of training samples
        """
        costs = deepcopy(prob) # Deepcopy probabilities

        # costs calculation
        if not isinstance(costs, pd.DataFrame):
            col_labels = np.arange(0, costs.shape[1])
            costs = pd.DataFrame(costs, columns=col_labels)
        for col in costs.columns:
            if not costs[col].between(0,1).all():
                raise ValueError(
                    f"Probabilities must be between 0 (inclusive) and 1 (inclusive)"
                )
        # budget calculation
        if threshold <= 0 or threshold > 1:
            raise ValueError(
                f"Threshold must be between 0 (exclusive) and 1 (inclusive)"
            )
        budget = -1 * costs.shape[0] * np.log(threshold)

        # Default probability of certainty of 1 as budget + 1
        conversion = lambda x : budget+1 if x==1 else -1 * np.log(1 - x)

        return costs.applymap(conversion), budget

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
        self.grb_model.model.optimize(mycallback)
        self.end_time = time.time()
        self.solving_time = self.end_time - self.start_time

        # Store fitted Gurobi model
        self.b_value = self.grb_model.model.getAttr("X", self.grb_model.b)
        self.w_value = self.grb_model.model.getAttr("X", self.grb_model.w)

        # `fit` should always return `self`
        return self

    def get_prediction(self, X):
        prediction = []
        for i in X.index:
            # Get prediction value
            node = 1
            while True:
                terminal = False
                for k in self.grb_model.labels:
                    if self.w_value[node, k] > 0.5:  # w[n,k] == 1
                        prediction += [k]
                        terminal = True
                        break
                if terminal:
                    break
                else:
                    for (f, theta) in self.grb_model.f_theta_indices:
                        if self.b_value[node, f, theta] > 0.5:  # b[n,f]== 1
                            if X.at[i, f] >= theta + 1:
                                node = self.grb_model.tree.get_right_children(node)
                            else:
                                node = self.grb_model.tree.get_left_children(node)
                            break
        return np.array(prediction)

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

        return self.get_prediction(df_test)

    def print_tree(self):
        """Print the fitted tree with the branching features, the threshold values for
        each branching node's test, and the predictions asserted for each assignment node"""

        check_is_fitted(self, ["grb_model"])

        assignment_nodes = []
        for n in self.grb_model.tree.Nodes + self.grb_model.tree.Leaves:
            print("#########node ", n)
            terminal = False

            # Check if pruned
            if self.grb_model.tree.get_parent(n) in assignment_nodes:
                print("pruned")
                assignment_nodes += [n]
                continue

            for k in self.grb_model.labels:
                if self.w_value[n, k] > 0.5:
                    print("leaf {}".format(k))
                    terminal = True
                    assignment_nodes += [n]
                    break
            if not terminal:
                for (f, theta) in self.grb_model.f_theta_indices:
                    if self.b_value[n, f, theta] > 0.5:  # b[n,f]== 1
                        print("Feature: ", f, ", Cutoff: ", theta)
                        break

    def plot_tree(
        self,
        label="all",
        filled=True,
        rounded=False,
        precision=3,
        ax=None,
        fontsize=None,
        color_dict={"node": None, "leaves": []},
    ):
        check_is_fitted(self, ["grb_model"])
        exporter = MPLPlotter(
            self.grb_model,
            self.X_col_labels,
            self.b_value,
            self.w_value,
            None,  # we will calculate p within get_node_status
            self.grb_model.tree.depth,
            self.classes_,
            label=label,
            filled=filled,
            rounded=rounded,
            precision=precision,
            fontsize=fontsize,
            color_dict=color_dict,
        )
        return exporter.export(ax=ax)
