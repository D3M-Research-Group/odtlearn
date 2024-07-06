from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y

from odtlearn import ODTL
from odtlearn.opt_ct import OptimalClassificationTree
from odtlearn.utils.callbacks import RobustBendersCallback
from odtlearn.utils.TreePlotter import MPLPlotter
from odtlearn.utils.validation import check_integer, check_same_as_X


class RobustOCT(OptimalClassificationTree):
    """An optimal robust decision tree classifier, fit on a given integer-valued
    data set and a given cost-and-budget uncertainty set to produce a tree robust
    against distribution shifts.

    Parameters
    ----------
    solver: str
        A string specifying the name of the solver to use
        to solve the MIP. Options are "Gurobi" and "CBC".
        If the CBC binaries are not found, Gurobi will be used by default.
    depth : int, default=1
        A parameter specifying the depth of the tree.
    time_limit : int, default=1800
        The given time limit for solving the MIP in seconds.
    num_threads: int, default=None
        The number of threads the solver should use. If not specified,
        solver uses all available threads.
    verbose : bool, default = False
        Flag for logging solver outputs.
    """

    def __init__(
        self,
        solver,
        depth=1,
        time_limit=1800,
        num_threads=None,
        verbose=False,
    ) -> None:
        super().__init__(
            solver,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

        # Regularization term: encourage less branching without sacrificing accuracy
        self._reg = 1 / (len(self._tree.Nodes) + 1)

    def _get_node_status(self, b, w, n):
        """
        This function give the status of a given node in a tree. By status we mean whether the node
            1- is pruned? i.e., we have made a prediction at one of its ancestors
            2- is a branching node? If yes, what feature do we branch on
            3- is a leaf? If yes, what is the prediction at this node?

        Parameters
        ----------
        model :
            The model solved to optimality (or reached to the time limit).
        b :
            The values of branching decision variable b.
        w :
            The values of prediction decision variable w.
        p :
            The values of decision variable p
        n :
            A valid node index in the tree

        Returns
        -------
        pruned : int
            pruned=1 iff the node is pruned
        branching : int
            branching = 1 iff the node branches at some feature f
        selected_feature : str
            The feature that the node branch on
        leaf : int
            leaf = 1 iff node n is a leaf in the tree
        value :  double
            if node n is a leaf, value represent the prediction at this node
        """
        pruned = False
        branching = False
        leaf = False
        value = None
        selected_feature = None
        cutoff = None

        p_sum = 0
        for m in self._tree.get_ancestors(n):
            # need to sum over all w values for a given n
            p_sum += sum(w[m, k] for k in self._labels)
        # to determine if a leaf, we look at its w value
        # and find for which label the value > 0.5
        col_idx = np.asarray([w[n, k] > 0.5 for k in self._labels]).nonzero()[0]
        # col_idx = np.asarray(w[n, :] > 0.5).nonzero().flatten()
        # assuming here that we can only have one column > 0.5
        if len(col_idx) > 0:
            leaf = True
            value = self._labels[int(col_idx[0])]
        elif p_sum == 1:
            pruned = True

        if not pruned and not leaf:
            for f, theta in self._f_theta_indices:
                if b[n, f, theta] > 0.5:
                    selected_feature = f
                    cutoff = theta
                    branching = True
        return pruned, branching, selected_feature, cutoff, leaf, value

    def _make_prediction(self, X):
        prediction = []
        for i in X.index:
            # Get prediction value
            node = 1
            while True:
                terminal = False
                for k in self._labels:
                    if self.w_value[node, k] > 0.5:  # w[n,k] == 1
                        prediction += [k]
                        terminal = True
                        break
                if terminal:
                    break
                else:
                    for f, theta in self._f_theta_indices:
                        if self.b_value[node, f, theta] > 0.5:  # b[n,f]== 1
                            if X.at[i, f] >= theta + 1:
                                node = self._tree.get_right_children(node)
                            else:
                                node = self._tree.get_left_children(node)
                            break
        return np.array(prediction)

    def _define_variables(self):
        # define variables

        # t is the objective value of the problem
        self._t = self._solver.add_vars(
            self._datapoints, vtype=ODTL.CONTINUOUS, ub=1, name="t"
        )
        # w[n,k] == 1 iff at node n we do not branch and we make the prediction k
        self._w = self._solver.add_vars(
            self._tree.Nodes + self._tree.Leaves,
            self._labels,
            vtype=ODTL.BINARY,
            name="w",
        )

        # b[n,f,theta] ==1 iff at node n we branch on feature f with cutoff theta
        self._b = self._solver.add_vars(self._b_indices, vtype=ODTL.BINARY, name="b")

    def _define_constraints(self):
        # define constraints

        # sum(b[n,f,theta], f, theta) + sum(w[n,k], k) = 1 for all n in nodes
        self._solver.add_constrs(
            (
                self._solver.quicksum(
                    self._b[n, f, theta] for (f, theta) in self._f_theta_indices
                )
                + self._solver.quicksum(self._w[n, k] for k in self._labels)
                == 1
            )
            for n in self._tree.Nodes
        )

        # sum(w[n,k], k) = 1 for all n in leaves
        self._solver.add_constrs(
            (self._solver.quicksum(self._w[n, k] for k in self._labels) == 1)
            for n in self._tree.Leaves
        )

    def _define_objective(self):
        # define objective function
        obj = self._solver.lin_expr(0)
        for i in self._datapoints:
            obj += self._t[i]
        # Add regularization term so that in case of tie in objective function,
        # encourage less branching
        obj += (
            -1
            * self._reg
            * self._solver.quicksum(
                self._b[n, f, theta] for (n, f, theta) in self._b_indices
            )
        )
        self._solver.set_objective(obj, ODTL.MAXIMIZE)

    def fit(self, X, y, costs=None, budget=-1):
        """
        Fit a robust optimal classification tree to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Features should be integer-valued.
        y : array-like of shape (n_samples,)
            The target values (class labels) for the training samples.
        costs : array-like of shape (n_samples, n_features), optional
            The costs of uncertainty for each feature and sample. If None, defaults to budget + 1.
        budget : float, optional
            The budget of uncertainty. Default is -1.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If X contains non-integer values or if inputs have inconsistent numbers of samples.

        Notes
        -----
        This method fits the RobustOCT model using mixed-integer optimization while
        considering potential adversarial perturbations within the given budget.
        It sets up the optimization problem, solves it, and stores the results.
        """
        self._extract_metadata(
            X,
            y,
        )
        X, y = check_X_y(X, y)
        check_integer(self._X)

        self._classes = unique_labels(y)

        self._cat_features = self._X_col_labels
        self._solver.store_data("cat_features", self._X_col_labels)

        # Get range of data, and store indices of branching variables based on range
        min_values = self._X.min(axis=0)
        max_values = self._X.max(axis=0)
        f_theta_indices = []
        b_indices = []
        for f in self._cat_features:
            min_value = min_values[f]
            max_value = max_values[f]

            # cutoffs are from min_value to max_value - 1
            for theta in range(min_value, max_value):
                f_theta_indices += [(f, theta)]
                b_indices += [(n, f, theta) for n in self._tree.Nodes]

        self._min_values = self._X.min(axis=0)
        self._max_values = self._X.max(axis=0)
        self._f_theta_indices = f_theta_indices
        self._b_indices = b_indices

        self._solver.store_data("min_values", self._X.min(axis=0))
        self._solver.store_data("max_values", self._X.max(axis=0))
        self._solver.store_data("f_theta_indices", self._f_theta_indices)
        self._solver.store_data("b_indices", self._b_indices)

        # Set default for costs of uncertainty if needed
        if costs is not None:
            self._costs = check_same_as_X(
                self._X, self._X_col_labels, costs, "uncertainty costs"
            )
            self._costs.set_index(pd.Index(range(costs.shape[0])), inplace=True)
            self._solver.store_data("costs", self._costs)
            # Also check if indices are the same
            if self._X.shape[0] != self._costs.shape[0]:
                raise ValueError(
                    (
                        f"Input covariates has {self._X.shape[0]} samples, "
                        f"but uncertainty costs has {self._costs.shape[0]}"
                    )
                )
        else:
            # By default, set costs to be budget + 1 (i.e. no uncertainty)
            gammas_df = deepcopy(self._X).astype("float")
            for col in gammas_df.columns:
                gammas_df[col].values[:] = budget + 1
            self._costs = gammas_df
            self._solver.store_data("costs", gammas_df)

        # Budget of uncertainty
        self._solver.store_data("budget", budget)
        self._budget = budget

        # Create uncertainty set
        self._epsilon = self._budget  # Budget of uncertainty
        self._gammas = self._costs  # Cost of feature uncertainty
        self._eta = self._budget + 1  # Cost of label uncertainty - future work
        self._solver.store_data("epsilon", self._epsilon)  # Budget of uncertainty
        self._solver.store_data("gammas", self._gammas)  # Cost of feature uncertainty
        self._solver.store_data(
            "eta", self._eta
        )  # Cost of label uncertainty - future work

        self._create_main_problem()

        # we need these in the callback to have access to the value of the decision variables for Gurobi callbacks
        self._solver.store_data("t", self._t)
        self._solver.store_data("b", self._b)
        self._solver.store_data("w", self._w)

        self._solver.store_data("solver", self._solver)
        self._solver.store_data("self", self)
        self._solver.store_data("datapoints", self._datapoints)

        callback_action = RobustBendersCallback

        self._solver.optimize(
            self._X,
            self,
            self._solver,
            callback=True,
            callback_action=callback_action,
            b=self._b,
            t=self._t,
            w=self._w,
        )

        # Store fitted decision variable values
        self.b_value = self._solver.get_var_value(self._b, "b")
        self.w_value = self._solver.get_var_value(self._w, "w")

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X using the fitted RobustOCT model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to make predictions. Features should be integer-valued.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for each sample in X.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If X contains non-integer values or has a different number of features than the training data.

        Notes
        -----
        This method uses the robust decision tree learned during the fit process to classify new samples.
        It traverses the tree for each sample in X, following the branching decisions until
        reaching a leaf node, and returns the corresponding class prediction.
        """
        check_is_fitted(self, ["b_value", "w_value"])
        # Convert to dataframe
        df_test = check_same_as_X(self._X, self._X_col_labels, X, "test covariates")
        check_integer(df_test)

        return self._make_prediction(df_test)

    def print_tree(self):
        check_is_fitted(self, ["b_value", "w_value"])
        assignment_nodes = []
        for n in self._tree.Nodes + self._tree.Leaves:
            print("#########node ", n)
            terminal = False

            # Check if pruned
            if self._tree.get_parent(n) in assignment_nodes:
                print("pruned")
                assignment_nodes += [n]
                continue

            for k in self._labels:
                if self.w_value[n, k] > 0.5:
                    print("leaf {}".format(k))
                    terminal = True
                    assignment_nodes += [n]
                    break
            if not terminal:
                for f, theta in self._solver.model._data["f_theta_indices"]:
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
        edge_annotation=True,
        arrow_annotation_font_scale=0.8,
        debug=False,
        feature_names=None,
    ):
        """
        Plot the fitted robust classification tree using matplotlib.

        Parameters
        ----------
        label : {'all', 'root', 'none'}, default='all'
            Whether to show informative labels for impurity, etc.
            Options include 'all' to show at every node, 'root' to show only at
            the top root node, or 'none' to not show at any node.
        filled : bool, default=True
            When set to True, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.
        rounded : bool, default=False
            When set to True, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.
        precision : int, default=3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        ax : matplotlib axis, default=None
            Axes to plot to. If None, use current axis. Any previous content
            is cleared.
        fontsize : int, default=None
            Size of text font. If None, determined automatically to fit figure.
        color_dict : dict, default={"node": None, "leaves": []}
            A dictionary specifying the colors for nodes and leaves in the plot in #RRGGBB format.
            If None, the colors are chosen using the sklearn `plot_tree` color palette.
        edge_annotation : bool, default=True
            Whether to display annotations on the edges.
        arrow_annotation_font_scale : float, default=0.8
            The font scale for the arrow annotations.
        debug : bool, default=False
            Whether to print debug information.
        feature_names : list of str, default=None
            A list of feature names to use for the plot. If None, the feature names from the
            fitted tree will be used. The feature names should be in the same order as the
            columns of the data used to fit the tree.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib Axes containing the plotted tree.

        Notes
        -----
        This method uses the MPLPlotter class to visualize the robust classification tree.
        """
        check_is_fitted(self, ["b_value", "w_value"])

        node_dict = {}
        for node in np.arange(1, self._tree.total_nodes + 1):
            node_dict[node] = self._get_node_status(self.b_value, self.w_value, node)

        # Use the provided feature names if available, otherwise use the original feature names
        if feature_names is not None:
            assert len(feature_names) == len(
                self._X_col_labels
            ), "The number of provided feature names does not match the number of columns in the data"
            column_names = feature_names
        else:
            column_names = self._X_col_labels

        exporter = MPLPlotter(
            self._tree,
            node_dict,
            column_names,
            self._tree.depth,
            self._classes,
            type(self).__name__,
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
