from copy import deepcopy

import numpy as np
import pandas as pd
from gurobipy import GRB, LinExpr, quicksum
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y

from odtlearn.opt_ct import OptimalClassificationTree
from odtlearn.utils.callbacks import robust_tree_callback
from odtlearn.utils.TreePlotter import MPLPlotter
from odtlearn.utils.validation import check_integer, check_same_as_X


class RobustOCT(OptimalClassificationTree):
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
    verbose : bool, default = True
        Flag for logging Gurobi outputs
    """

    def __init__(
        self,
        depth=1,
        time_limit=1800,
        num_threads=None,
        verbose=False,
    ) -> None:
        self.model_name = "RobustOCT"
        super().__init__(
            depth,
            time_limit,
            num_threads,
            verbose,
        )

        # Regularization term: encourage less branching without sacrificing accuracy
        self.reg = 1 / (len(self._tree.Nodes) + 1)

        # Decision Variables (b,w inherited from problem formulation)
        self._t = 0
        # The cuts we add in the callback function would be treated as lazy constraints
        self._model.params.LazyConstraints = 1

        """
        The following variables are used for the Benders problem to keep track of the times we call the callback.

        - counter_integer tracks number of times we call the callback from an integer node
         in the branch-&-bound tree
            - time_integer tracks the associated time spent in the callback for these calls
        - counter_general tracks number of times we call the callback from a non-integer node
         in the branch-&-bound tree
            - time_general tracks the associated time spent in the callback for these calls

        the ones ending with success are related to success calls. By success we mean ending
        up adding a lazy constraint to the model
        """
        self._model._total_callback_time_integer = 0
        self._model._total_callback_time_integer_success = 0

        self._model._total_callback_time_general = 0
        self._model._total_callback_time_general_success = 0

        self._model._callback_counter_integer = 0
        self._model._callback_counter_integer_success = 0

        self._model._callback_counter_general = 0
        self._model._callback_counter_general_success = 0

        self._model._total_cuts = 0

        # We also pass the following information to the model as we need them in the callback
        self._model._master = self

    def _get_node_status(self, b, w, n):
        """
        This function give the status of a given node in a tree. By status we mean whether the node
            1- is pruned? i.e., we have made a prediction at one of its ancestors
            2- is a branching node? If yes, what feature do we branch on
            3- is a leaf? If yes, what is the prediction at this node?

        Parameters
        ----------
        grb_model :
            The gurobi model solved to optimality (or reached to the time limit).
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
                    for (f, theta) in self._f_theta_indices:
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
        self._t = self._model.addVars(
            self._datapoints, vtype=GRB.CONTINUOUS, ub=1, name="t"
        )
        # w[n,k] == 1 iff at node n we do not branch and we make the prediction k
        self._w = self._model.addVars(
            self._tree.Nodes + self._tree.Leaves,
            self._labels,
            vtype=GRB.BINARY,
            name="w",
        )

        # b[n,f,theta] ==1 iff at node n we branch on feature f with cutoff theta
        self._b = self._model.addVars(self._b_indices, vtype=GRB.BINARY, name="b")

        # we need these in the callback to have access to the value of the decision variables
        self._model._vars_t = self._t
        self._model._vars_b = self._b
        self._model._vars_w = self._w

    def _define_constraints(self):
        # define constraints

        # sum(b[n,f,theta], f, theta) + sum(w[n,k], k) = 1 for all n in nodes
        self._model.addConstrs(
            (
                quicksum(self._b[n, f, theta] for (f, theta) in self._f_theta_indices)
                + quicksum(self._w[n, k] for k in self._labels)
                == 1
            )
            for n in self._tree.Nodes
        )

        # sum(w[n,k], k) = 1 for all n in leaves
        self._model.addConstrs(
            (quicksum(self._w[n, k] for k in self._labels) == 1)
            for n in self._tree.Leaves
        )

    def _define_objective(self):
        # define objective function
        obj = LinExpr(0)
        for i in self._datapoints:
            obj.add(self._t[i])
        # Add regularization term so that in case of tie in objective function,
        # encourage less branching
        obj.add(
            -1
            * self.reg
            * quicksum(self._b[n, f, theta] for (n, f, theta) in self._b_indices)
        )

        self._model.setObjective(obj, GRB.MAXIMIZE)

    def fit(self, X, y, costs=None, budget=-1):
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

        Returns
        -------
        self : object
            Returns self.
        """
        self._extract_metadata(
            X,
            y,
        )
        X, y = check_X_y(X, y)
        check_integer(self._X)

        self._classes = unique_labels(y)

        self._cat_features = self._X_col_labels

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

        # Set default for costs of uncertainty if needed
        if costs is not None:
            self._costs = check_same_as_X(
                self._X, self._X_col_labels, costs, "uncertainty costs"
            )
            self._costs.set_index(pd.Index(range(costs.shape[0])), inplace=True)
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

        # Budget of uncertainty
        self._budget = budget

        # Create uncertainty set
        self._epsilon = self._budget  # Budget of uncertainty
        self._gammas = self._costs  # Cost of feature uncertainty
        self._eta = self._budget + 1  # Cost of label uncertainty - future work

        self._create_main_problem()
        self._model.update()
        self._model.optimize(robust_tree_callback)

        # Store fitted Gurobi model
        self.b_value = self._model.getAttr("X", self._b)
        self.w_value = self._model.getAttr("X", self._w)

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
                for (f, theta) in self._f_theta_indices:
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
    ):
        """Plot the fitted tree with the branching features, the threshold values for
        each branching node's test, and the predictions asserted for each assignment node
        using matplotlib. The method uses the Gurobi model's name for determining how
        to generate the tree. It does some preprocessing before passing the tree to the
        `_MPLTreeExporter` class from the sklearn package. The arguments for the
        `plot_tree` method are based on the arguments of the sklearn `plot_tree` function.

        Parameters
        ----------
        label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

        filled : bool, default=False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        rounded : bool, default=False
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        precision: int, default=3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        ax : matplotlib axis, default=None
            Axes to plot to. If None, use current axis. Any previous content
        is cleared.

        fontsize : int, default=None
            Size of text font. If None, determined automatically to fit figure.

        color_dict: dict, default={"node": None, "leaves": []}
            A dictionary specifying the colors for nodes and leaves in the plot in #RRGGBB format.
            If None, the colors are chosen using the sklearn `plot_tree` color palette
        """
        check_is_fitted(self, ["b_value", "w_value"])

        node_dict = {}
        for node in np.arange(1, self._tree.total_nodes + 1):
            node_dict[node] = self._get_node_status(self.b_value, self.w_value, node)

        exporter = MPLPlotter(
            self._tree,
            node_dict,
            self._X_col_labels,
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
