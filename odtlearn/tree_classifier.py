import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from odtlearn.utils.TreePlotter import MPLPlotter


class TreeClassifier(ClassifierMixin, BaseEstimator):
    """A base class for all other classifiers in the ODTlearn package
    that contains some universal methods for preprocessing data as well as
    printing and plotting fitted optimal decision trees.
    """

    def __init__(self, depth=1, time_limit=60, num_threads=None) -> None:
        self.depth = depth
        self.time_limit = time_limit
        self.num_threads = num_threads

    def extract_metadata(self, X, y, **kwargs):
        """A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API

        Additional variables that need to be processed for
        different classifier types are passed through keyword arguments
        """
        if isinstance(X, pd.DataFrame):
            self.X_col_labels = X.columns
            self.X_col_dtypes = X.dtypes
            self.X = X
        else:
            self.X_col_labels = np.array([f"X_{i}" for i in np.arange(0, X.shape[1])])
            self.X = pd.DataFrame(X, columns=self.X_col_labels)

        # Strip indices in training data into integers
        self.X.set_index(pd.Index(range(X.shape[0])), inplace=True)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.y = y.values
        else:
            self.y = y
        self.labels = np.unique(y)

        if kwargs.get("protect_feat", None) is not None:
            protect_feat = kwargs.get("protect_feat")
            if isinstance(protect_feat, pd.DataFrame):
                self.protect_feat_col_labels = protect_feat.columns
                self.protect_feat_col_dtypes = protect_feat.dtypes
            else:
                self.protect_feat_col_labels = np.array(
                    [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
                )
        if kwargs.get("t", None) is not None:
            t = kwargs.get("t")
            self.treatments = np.unique(t)

    def get_node_status(self, grb_model, b, w, p, n):
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

        model_type = grb_model.model.ModelName
        assert len(model_type) > 0

        # Node status for StrongTree and FairTree
        if model_type in ["FairOCT", "FlowOCT", "BendersOCT", "FlowOPT", "IPW"]:
            cutoff = 0
            p_sum = 0
            for m in grb_model.tree.get_ancestors(n):
                p_sum = p_sum + p[m]
            if p[n] > 0.5:  # leaf
                leaf = True
                labels = (
                    grb_model.treatments_set
                    if model_type in ["FlowOPT", "IPW"]
                    else grb_model.labels
                )
                for k in labels:
                    if w[n, k] > 0.5:
                        value = k
            elif p_sum == 1:  # Pruned
                pruned = True

            if n in grb_model.tree.Nodes:
                if (pruned is False) and (leaf is False):  # branching
                    for f in grb_model.X_col_labels:
                        if b[n, f] > 0.5:
                            selected_feature = f
                            branching = True
        elif model_type == "RobustOCT":
            p_sum = 0
            for m in grb_model.tree.get_ancestors(n):
                # need to sum over all w values for a given n
                p_sum += sum(w[m, k] for k in grb_model.labels)
            # to determine if a leaf, we look at its w value
            # and find for which label the value > 0.5
            col_idx = np.asarray([w[n, k] > 0.5 for k in grb_model.labels]).nonzero()[0]
            # col_idx = np.asarray(w[n, :] > 0.5).nonzero().flatten()
            # assuming here that we can only have one column > 0.5
            if len(col_idx) > 0:
                leaf = True
                value = grb_model.labels[int(col_idx[0])]
            elif p_sum == 1:
                pruned = True

            if not pruned and not leaf:
                for f, theta in grb_model.f_theta_indices:
                    if b[n, f, theta] > 0.5:
                        selected_feature = f
                        cutoff = theta
                        branching = True

        return pruned, branching, selected_feature, cutoff, leaf, value

    def print_tree(self):
        """Print the fitted tree with the branching features, the threshold values for
        each branching node's test, and the predictions asserted for each assignment node

        The method uses the Gurobi model's name for determining how to generate the tree
        """
        check_is_fitted(self, ["grb_model"])
        if self.grb_model.model.ModelName in [
            "FlowOCT",
            "FairOCT",
            "BendersOCT",
            "FlowOPT",
            "IPW",
        ]:
            for n in self.grb_model.tree.Nodes + self.grb_model.tree.Leaves:
                (
                    pruned,
                    branching,
                    selected_feature,
                    _,
                    leaf,
                    value,
                ) = self.get_node_status(
                    self.grb_model, self.b_value, self.w_value, self.p_value, n
                )
                print("#########node ", n)
                if pruned:
                    print("pruned")
                elif branching:
                    print("branch on {}".format(selected_feature))
                elif leaf:
                    print("leaf {}".format(value))
        elif self.grb_model.model.ModelName in ["RobustOCT"]:
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

    def _get_prediction(self, X):
        prediction = []
        if self.grb_model.model.ModelName in [
            "FairOCT",
            "FlowOCT",
            "BendersOCT",
            "FlowOPT",
            "IPW",
        ]:
            for i in range(X.shape[0]):
                current = 1
                while True:
                    (
                        _,
                        branching,
                        selected_feature,
                        _,
                        leaf,
                        value,
                    ) = self.get_node_status(
                        self.grb_model,
                        self.b_value,
                        self.w_value,
                        self.p_value,
                        current,
                    )
                    if leaf:
                        prediction.append(value)
                        break
                    elif branching:
                        selected_feature_idx = np.where(
                            self.grb_model.X_col_labels == selected_feature
                        )
                        # Raise assertion error we don't have a column that matches
                        # the selected feature or more than one column that matches
                        assert (
                            len(selected_feature_idx) == 1
                        ), f"Found {len(selected_feature_idx)} columns matching the selected feature {selected_feature}"
                        if X[i, selected_feature_idx] == 1:  # going right on the branch
                            current = self.grb_model.tree.get_right_children(current)
                        else:  # going left on the branch
                            current = self.grb_model.tree.get_left_children(current)
        elif self.grb_model.model.ModelName in ["RobustOCT"]:
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
        check_is_fitted(self, ["grb_model"])
        print(f"color dict in tree_classifier: {color_dict}")
        exporter = MPLPlotter(
            self.grb_model,
            self.X_col_labels,
            self.b_value,
            self.w_value,
            None if self.grb_model.model.ModelName in ["RobustOCT"] else self.p_value,
            self.grb_model.tree.depth,
            self.treatments
            if self.grb_model.model.ModelName in ["FlowOPT", "IPW"]
            else self.classes_,
            self.get_node_status,
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
