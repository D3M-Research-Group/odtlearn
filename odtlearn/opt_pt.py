import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from odtlearn.opt_dt import OptimalDecisionTree
from odtlearn.utils.TreePlotter import MPLPlotter


class OptimalPrescriptiveTree(OptimalDecisionTree):
    def __init__(
        self,
        solver,
        depth,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        super().__init__(solver, depth, time_limit, num_threads, verbose)

    def _extract_metadata(self, X, y, t):
        """A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API

        Additional variables that need to be processed for
        different classifier types are passed through keyword arguments
        """
        if isinstance(X, pd.DataFrame):
            self._X_col_labels = X.columns
            self._X_col_dtypes = X.dtypes
            self._X = X
        else:
            self._X_col_labels = np.array([f"X_{i}" for i in np.arange(0, X.shape[1])])
            self._X = pd.DataFrame(X, columns=self._X_col_labels)

        # Strip indices in training data into integers
        self._X.set_index(pd.Index(range(self._X.shape[0])), inplace=True)
        self._datapoints = np.arange(0, self._X.shape[0])

        if isinstance(y, (pd.Series, pd.DataFrame)):
            self._y = y.values
        else:
            self._y = y
        self._labels = np.unique(self._y)

        self._t = t
        self._treatments = np.unique(t)

    def _get_node_status(self, b, w, p, n):
        """
        This function give the status of a given node in a tree. By status we mean whether the node
        1- is pruned? i.e., we have made a prediction at one of its ancestors
        2- is a branching node? If yes, what feature do we branch on
        3- is a leaf? If yes, what is the prediction at this node?

        Parameters
        ----------
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

        cutoff = 0
        p_sum = 0
        for m in self._tree.get_ancestors(n):
            p_sum = p_sum + p[m]
        if p[n] > 0.5:  # leaf
            leaf = True
            labels = self._treatments
            for k in labels:
                if w[n, k] > 0.5:
                    value = k
        elif p_sum == 1:  # Pruned
            pruned = True

        if n in self._tree.Nodes:
            if (pruned is False) and (leaf is False):  # branching
                for f in self._X_col_labels:
                    if b[n, f] > 0.5:
                        selected_feature = f
                        branching = True
        return pruned, branching, selected_feature, cutoff, leaf, value

    def _make_prediction(self, X):
        prediction = []
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
                ) = self._get_node_status(
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
                        self._X_col_labels == selected_feature
                    )
                    # Raise assertion error we don't have a column that matches
                    # the selected feature or more than one column that matches
                    assert (
                        len(selected_feature_idx) == 1
                    ), f"Found {len(selected_feature_idx)} columns matching the selected feature {selected_feature}"
                    if X[i, selected_feature_idx] == 1:  # going right on the branch
                        current = self._tree.get_right_children(current)
                    else:  # going left on the branch
                        current = self._tree.get_left_children(current)
        return np.array(prediction)

    def print_tree(self):
        """Print the fitted tree with the branching features, the threshold values for
        each branching node's test, and the predictions asserted for each assignment node

        The method uses the Gurobi model's name for determining how to generate the tree
        """
        check_is_fitted(self, ["b_value", "w_value", "p_value"])
        for n in self._tree.Nodes + self._tree.Leaves:
            (
                pruned,
                branching,
                selected_feature,
                _,
                leaf,
                value,
            ) = self._get_node_status(self.b_value, self.w_value, self.p_value, n)
            print("#########node ", n)
            if pruned:
                print("pruned")
            elif branching:
                print("branch on {}".format(selected_feature))
            elif leaf:
                print("leaf {}".format(value))

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
        distance=1.0,
    ):
        """
        Plot the fitted tree with the branching features, the threshold values for
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
        check_is_fitted(self, ["b_value", "w_value", "p_value"])

        node_dict = {}
        for node in np.arange(1, self._tree.total_nodes + 1):
            node_dict[node] = self._get_node_status(
                self.b_value, self.w_value, self.p_value, node
            )

        exporter = MPLPlotter(
            self._tree,
            node_dict,
            self._X_col_labels,
            self._tree.depth,
            self._treatments,
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
        return exporter.export(ax=ax, distance=distance)
