from typing import Union, Optional

import numpy as np
from numpy import ndarray, int64, str_
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from odtlearn import ODTL
from odtlearn.opt_ct import OptimalClassificationTree
from odtlearn.utils.validation import check_columns_match


class BertsimasDunnOCT(OptimalClassificationTree):
    """
    An optimal decision tree classifier based on the OCT formulation of Bertsimas and Dunn (2017)

    Parameters
    ----------
    solver : str
        The solver to use for the MIP formulation. Currently, only "gurobi" and "CBC" are supported.
    alpha : float, default=0
        The regularization parameter for controlling the complexity of the learned tree.
    N_min : int, default=1
        The minimum number of datapoints in a leaf
    depth : int, default=1
        The maximum depth of the tree to be learned.
    time_limit : int, default=60
        The time limit (in seconds) for solving the MIP formulation.
    num_threads : int, default=None
        The number of threads the solver should use. If not specified,
        solver uses all available threads
    verbose : bool, default=False
        Whether to print verbose output during the tree learning process.

    Methods
    -------
    fit(X, y)
        Fit the optimal classification tree to the given training data.
    predict(X)
        Make predictions using the fitted optimal classification tree.
    _define_variables()
        Define the variables for the optimization problem.
    _define_constraints()
        Define the constraints for the optimization problem.
    _define_objective()
        Define the objective function for the optimization problem.

    Notes
    -----
    This implementation is faithful to the Bertsimas and Dunn (2017) formulation, with the only
    change being an additional constraint to ensure that internal prediction nodes have their prediction
    in the right-most subtree, per the convention in Bertsimas and Dunn.
    """

    def __init__(
        self,
        solver: str,
        alpha: float = 0,
        N_min: int = 1,
        depth: int = 1,
        time_limit: int = 60,
        num_threads: Union[None, int] = None,
        verbose: bool = False,
    ) -> None:
        self._N_min = N_min  # Minimum number in each leaf
        self._alpha = alpha  # regularization parameter
        super().__init__(
            solver,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def fit(
        self,
        X: Union[ndarray, DataFrame],
        y: Union[ndarray, Series],
    ) -> "BertsimasDunnOCT":
        """
        Fit the BertsimasDunnOCT model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each feature should be binary (0 or 1).
        y : array-like of shape (n_samples,)
            The target values (class labels) for the training samples.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If X contains non-binary values, or if X and y have inconsistent numbers of samples.
            Also raised if weights are not provided when obj_mode is 'weighted', or if the number
            of weights doesn't match the number of samples.

        Notes
        -----
        The behavior of this method depends on the `obj_mode` specified during initialization:
        - If obj_mode is 'acc', equal weights are used (weights parameter is ignored).
        - If obj_mode is 'balance', weights are automatically calculated to balance class importance.
        - If obj_mode is 'weighted', the provided weights are used.

        When obj_mode is not 'weighted' and weights are provided, a warning is issued and the weights are ignored.

        This method fits the FlowOCT model using mixed-integer optimization.
        It sets up the optimization problem, solves it, and stores the results.
        """

        # extract column labels, unique classes and store X as a DataFrame
        self._extract_metadata(X, y)

        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)

        # Check that the features are normalized
        if not ((X >= 0) & (X <= 1)).all().all():
            raise ValueError("All values in X must be between 0 and 1 inclusive.")

        # Store the classes seen during fit
        self._classes = unique_labels(y)

        self._create_main_problem()
        self._solver.optimize()

        self.L_value = self._solver.get_var_value(self._L, "L")
        self.d_value = self._solver.get_var_value(self._d, "d")
        self.N_value = self._solver.get_var_value(self._Nt, "Nt")
        self.Nkt_value = self._solver.get_var_value(self._Nkt, "Nkt")
        self.c_value = self._solver.get_var_value(self._c, "c")
        self.z_value = self._solver.get_var_value(self._z, "z")
        self.l_value = self._solver.get_var_value(self._l, "l")
        self.a_value = self._solver.get_var_value(self._a, "a")
        self.b_value = self._solver.get_var_value(self._b, "b")

        return self

    def predict(self, X: Union[DataFrame, ndarray]) -> ndarray:
        """
        Predict class labels for samples in X using the fitted FlowOCT model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to make predictions. The features should
            match those used during the fit method.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for each sample in X.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.

        ValueError
            If the input X has a different number of features than the training data.

        Notes
        -----
        This method uses the decision tree learned during the fit process to classify
        new samples. It traverses the tree for each sample in X, following the branching
        decisions until reaching a leaf node, and returns the corresponding class prediction.

        The input X should have the same feature set as the training data used in fit().
        If categorical variables were one-hot encoded for training, the same encoding
        should be applied to X before calling predict().

        Examples
        --------
        >>> from odtlearn.flow_oct import FlowOCT
        >>> import numpy as np
        >>> X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        >>> y_train = np.array([0, 1, 1, 0])
        >>> clf = FlowOCT(depth=2)
        >>> clf.fit(X_train, y_train)
        >>> X_test = np.array([[1, 1], [0, 0]])
        >>> y_pred = clf.predict(X_test)
        >>> print(y_pred)
        [1 0]
        """
        # Check is fit had been called
        # for now we are assuming the model has been fit successfully if the fitted values for b, w, and p exist
        check_is_fitted(self, ["a_value", "b_value", "c_value", "d_value"])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        # Check that the features are normalized
        if not ((X >= 0) & (X <= 1)).all().all():
            raise ValueError("All values in X must be between 0 and 1 inclusive.")

        check_columns_match(self._X_col_labels, X)

        return self._make_prediction(X)

    def _define_variables(self):
        self._L = self._solver.add_vars(
            self._tree.Leaves, vtype=ODTL.CONTINUOUS, lb=0, name="L"
        )
        self._d = self._solver.add_vars(self._tree.Nodes, vtype=ODTL.BINARY, name="d")
        self._Nt = self._solver.add_vars(
            self._tree.Leaves, vtype=ODTL.CONTINUOUS, lb=0, name="Nt"
        )
        self._Nkt = self._solver.add_vars(
            self._labels, self._tree.Leaves, vtype=ODTL.CONTINUOUS, lb=0, name="Nkt"
        )
        self._c = self._solver.add_vars(
            self._labels, self._tree.Leaves, vtype=ODTL.BINARY, lb=0, name="c"
        )
        self._z = self._solver.add_vars(
            self._datapoints, self._tree.Leaves, vtype=ODTL.BINARY, name="z"
        )
        self._l = self._solver.add_vars(self._tree.Leaves, vtype=ODTL.BINARY, name="l")
        self._a = self._solver.add_vars(
            self._X_col_labels, self._tree.Nodes, vtype=ODTL.BINARY, name="a"
        )
        self._b = self._solver.add_vars(
            self._tree.Nodes, vtype=ODTL.CONTINUOUS, lb=0, name="b"
        )

    def _define_constraints(self):
        # Quantities to be calculated for the formulation
        Yik = self._define_Yik()
        epsilon = self._define_epsilon()
        epsilon_max = max(epsilon.values())

        # Constraint 1
        self._solver.add_constrs(
            (
                self._L[t]
                >= self._Nt[t]
                - self._Nkt[k, t]
                - (len(self._datapoints) * (1 - self._c[k, t]))
            )
            for k in self._labels
            for t in self._tree.Leaves
        )

        # Constraint 2
        self._solver.add_constrs(
            (
                self._L[t]
                <= self._Nt[t]
                - self._Nkt[k, t]
                + (len(self._datapoints) * self._c[k, t])
            )
            for k in self._labels
            for t in self._tree.Leaves
        )

        # constraint 4
        self._solver.add_constrs(
            (
                self._Nkt[k, t]
                == self._solver.quicksum(
                    (0.5 * (1 + Yik[i, k]) * self._z[i, t]) for i in self._datapoints
                )
            )
            for k in self._labels
            for t in self._tree.Leaves
        )

        # constraint 5
        self._solver.add_constrs(
            (
                self._Nt[t]
                == self._solver.quicksum(self._z[i, t] for i in self._datapoints)
            )
            for t in self._tree.Leaves
        )

        # constraint 6
        self._solver.add_constrs(
            (self._l[t] == self._solver.quicksum(self._c[k, t] for k in self._labels))
            for t in self._tree.Leaves
        )
        big_M1 = 1
        big_M2 = 1 + epsilon_max
        for t in self._tree.Leaves:
            # Constraint 7
            for m in self._tree.get_right_ancestors(t):
                self._solver.add_constrs(
                    (
                        self._solver.quicksum(
                            (self._a[j, m] * self._X.at[i, j])
                            for j in self._X_col_labels
                        )
                        >= self._b[m] - (big_M1 * (1 - self._z[i, t]))
                    )
                    for i in self._datapoints
                )
            # Constraint 8
            for m in self._tree.get_left_ancestors(t):
                self._solver.add_constrs(
                    (
                        self._solver.quicksum(
                            (self._a[j, m] * (self._X.at[i, j] + epsilon[j]))
                            for j in self._X_col_labels
                        )
                        <= self._b[m] + (big_M2 * (1 - self._z[i, t]))
                    )
                    for i in self._datapoints
                )

        # constraint 9
        self._solver.add_constrs(
            (self._solver.quicksum((self._z[i, t]) for t in self._tree.Leaves) == 1)
            for i in self._datapoints
        )

        # Constraint 10
        self._solver.add_constrs(
            (self._z[i, t] <= self._l[t])
            for i in self._datapoints
            for t in self._tree.Leaves
        )

        # ADDED CONSTRAINT - NOT IN ORIGINAL FORMULATION
        # Enforces that if a node is pruned, all samples must end up in the right-most leaf in the subtree
        for t in self._tree.Leaves:
            for m in self._tree.get_left_ancestors(t):
                self._solver.add_constrs(
                    (self._z[i, t] <= self._d[m]) for i in self._datapoints
                )

        # Constraint 11
        self._solver.add_constrs(
            (
                self._solver.quicksum((self._z[i, t]) for i in self._datapoints)
                >= self._N_min * self._l[t]
            )
            for t in self._tree.Leaves
        )

        # constraint 12
        self._solver.add_constrs(
            (
                self._d[t]
                == self._solver.quicksum((self._a[j, t]) for j in self._X_col_labels)
            )
            for t in self._tree.Nodes
        )

        # Constraint 13
        self._solver.add_constrs((self._b[t] <= self._d[t]) for t in self._tree.Nodes)

        # Constraint 14
        for t in self._tree.Nodes:
            if t != 1:
                self._solver.add_constr(self._d[t] <= self._d[self._tree.get_parent(t)])

    def _define_objective(self):
        obj = self._solver.lin_expr()

        _, count = np.unique(self._y, return_counts=True)
        L_hat = len(self._y) - count.max()

        obj.add(
            self._solver.quicksum((1 / L_hat) * self._L[t] for t in self._tree.Leaves)
        )
        obj.add(
            self._solver.quicksum(self._alpha * self._d[t] for t in self._tree.Nodes)
        )

        self._solver.set_objective(obj, ODTL.MINIMIZE)

    def _get_node_status(
        self,
        a: dict[int, int],
        b: dict[int],
        c: dict[int, int],
        d: dict[int],
        n: Union[int, int64],
        feature_names: Optional[ndarray] = None,
    ) -> Union[
        tuple[bool, bool, None, int, bool, int64],
        tuple[bool, bool, str_, int, bool, None],
    ]:

        if n in self._tree.Nodes and d[n] > 0.5:
            for feat_idx, f in enumerate(self._X_col_labels):
                if a[f, n] > 0.5:
                    selected_feature = (
                        f if feature_names is None else feature_names[feat_idx]
                    )
                    return False, True, selected_feature, b[n], False, None

        if n != 1 and d[self._tree.get_parent(n)] == 0:
            # Parent doesn't branch = pruned
            return True, False, None, None, False, None

        # Otherwise, leaf
        label_node = n
        if n in self._tree.Nodes:
            while label_node in self._tree.Nodes:
                label_node = self._tree.get_right_children(label_node)

        for k in self._labels:
            if c[k, label_node] > 0.5:
                return False, False, None, None, True, k

        raise RuntimeError(f"No status found for node {n}")

    def _make_prediction(self, X: ndarray) -> ndarray:
        prediction = []
        for i in range(X.shape[0]):
            n = 1
            while True:
                (
                    _,
                    branching,
                    selected_feature,
                    cutoff,
                    leaf,
                    value,
                ) = self._get_node_status(
                    self.a_value,
                    self.b_value,
                    self.c_value,
                    self.d_value,
                    n,
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
                    if (
                        X[i, selected_feature_idx] >= cutoff
                    ):  # going right on the branch
                        n = self._tree.get_right_children(n)
                    else:  # going left on the branch
                        n = self._tree.get_left_children(n)
        return np.array(prediction)

    def _define_Yik(self):
        """
        Construct the Y_{ik} dictionary, where:
            Y_{ik} = +1 if sample i has label k,
                    -1 otherwise.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        dict
            Dictionary indexed by (i, k), where i is the sample index
            and k is a class label.
        """
        labels = sorted(set(self._y))

        Yik = {}
        for i, yi in enumerate(self._y):
            for k in labels:
                Yik[(i, k)] = 1 if yi == k else -1

        return Yik

    def _define_epsilon(self):
        """
        Compute the smallest non-zero difference for each feature.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        Returns
        -------
        dict
            Dictionary mapping each feature name to its smallest non-zero
            difference. If a feature has only one unique value, its epsilon
            is set to 0.0.
        """
        epsilon = {}

        for col in self._X_col_labels:
            # Get sorted unique values
            values = sorted(self._X[col].unique())

            if len(values) < 2:
                epsilon[col] = 0.0
                continue

            # Compute adjacent differences
            diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

            epsilon[col] = min(diffs)

        return epsilon

    def print_tree(self) -> None:
        """
        Print a text representation of the fitted tree.

        This method prints the structure of the fitted tree, including the branching features,
        the threshold values for each branching node's test, and the predictions asserted for each leaf node.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.

        Notes
        -----
        The tree is printed in a depth-first manner, with each node represented by its index,
        branching feature and threshold (for internal nodes), or prediction (for leaf nodes).
        """
        check_is_fitted(self, ["a_value", "b_value", "c_value", "d_value"])
        for n in self._tree.Nodes + self._tree.Leaves:
            (
                pruned,
                branching,
                selected_feature,
                cutoff,
                leaf,
                value,
            ) = self._get_node_status(
                self.a_value,
                self.b_value,
                self.c_value,
                self.d_value,
                n,
            )
            print("#########node ", n)
            if pruned:
                print("pruned")
            elif branching:
                print("branch on {}, threshold={}".format(selected_feature, cutoff))
            elif leaf:  # pragma: no cover
                # a tree will always have leaves
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
        feature_names=None,
    ):
        """
        Plot the fitted tree using matplotlib.

        Parameters
        ----------
        label : {'all', 'root', 'none'}, default='all'
            Whether to show informative labels for impurity, etc.
            Options include 'all' to show at every node, 'root' to show only at
            the top root node, or 'none' to not show at any node.
        filled : bool, default=True
            When set to True, paint nodes to indicate majority treatment for
            prescriptive trees.
        rounded : bool, default=False
            When set to True, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.
        precision : int, default=3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        ax : matplotlib axis, default=None
            Axes to plot to. If None, use current axis. Any previous content is cleared.
        fontsize : int, default=None
            Size of text font. If None, determined automatically to fit figure.
        color_dict : dict, optional
            A dictionary specifying the colors for nodes and leaves.
            Default: {"node": None, "leaves": []}
        edge_annotation : bool, optional (default=True)
            Whether to display annotations on the edges.
        arrow_annotation_font_scale : float, optional (default=0.5)
            The font scale for the arrow annotations.
        debug : bool, optional (default=False)
            Whether to print debug information.
        distance: float, default=1.0
            Adjust distance between levels in the tree.
        feature_names : list of str, default=None
            A list of feature names to use for the plot. If None, the feature names from the
            fitted tree will be used. The feature names should be in the same order as the
            columns of the data used to fit the tree.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib Axes containing the plot.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.

        Notes
        -----
        This method visualizes the fitted tree structure using matplotlib.
        Each node in the tree is represented by a box, with arrows indicating the branching structure.
        """
        check_is_fitted(self, ["a_value", "b_value", "c_value", "d_value"])

        # Use the provided feature names if available, otherwise use the original feature names
        if feature_names is not None:
            assert len(feature_names) == len(
                self._X_col_labels
            ), "The number of provided feature names does not match the number of columns in the data"
            column_names = feature_names
        else:
            column_names = self._X_col_labels

        node_dict = {}
        for node in np.arange(1, self._tree.total_nodes + 1):
            node_dict[node] = self._get_node_status(
                self.a_value,
                self.b_value,
                self.c_value,
                self.d_value,
                node,
                feature_names=column_names,
            )
        try:
            from odtlearn.utils.TreePlotter import MPLPlotter

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
        except ModuleNotFoundError as e:
            raise ImportError(
                "Plotting requires the 'seaborn' package, which is not installed. "
                "Install it with: [pip install seaborn]. "
            ) from e
        return exporter.export(ax=ax, distance=distance)

    def plot_search_progress(
        self,
        ax=None,
        title="Search Progress",
        xlabel="Time (s)",
        ylabel="Objective Bound",
        lw=2,
        alpha=0.8,
        legend_loc="best",
        legend_fontsize=None,
        grid=True,
        color_ub="b",
        color_lb="r",
        log_scale=False,
    ) -> "matplotlib.axes.Axes":  # noqa: F821
        """
        Plots the search progress log showing the lower and upper objective bounds over time.

        Parameters
        ----------
        ax : matplotlib axis, default=None
            Axes to plot to. If None, use current axis. Any previous content is cleared.
        title : str, optional, default: 'Search Progress'
            The title of the plot.
        xlabel : str, optional, default: 'Time (s)'
            The label for the x-axis.
        ylabel : str, optional, default: 'Objective Bound'
            The label for the y-axis.
        lw : float, optional, default: 2
            The line width of the bound lines.
        alpha : float, optional, default: 0.8
            The alpha blending value between 0 (transparent) and 1 (opaque).
        legend_loc : str, optional, default: 'best'
            The location of the legend.
        legend_fontsize : int or str, optional
            The font size for the legend labels.
        grid : bool, optional, default: True
            Whether to show the grid lines.
        color_ub : color str or tuple of floats, optional, default: 'b'
            The color to use for the upper bound line.
        color_lb : color str or tuple of floats, optional, default: 'r'
            The color to use for the lower bound line.
        log_scale : bool, optional, default: False
            Whether to use a log scale for the y-axis.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        AttributeError
            If the search progress log was not recorded during fitting.

        Notes
        -----
        The search progress log must be enabled prior to fitting by setting
        `store_search_progress_log` to True.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Plotting requires the 'matplotlib' package, which is not installed. "
                "Install it with: [pip install matplotlib]. "
            ) from e

        # Check if model has been fit
        check_is_fitted(self, ["a_value", "b_value", "c_value", "d_value"])

        # Check if search progress log exists
        if len(self.search_progress_log) == 0:
            raise AttributeError(
                "No search progress log found. Make sure to set "
                "'store_search_progress_log=True' before fitting."
            )

        # Extract times and bounds from log
        times, bounds = zip(*self.search_progress_log)
        lb, ub = zip(*bounds)

        # Create plot
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.plot(times, lb, label="Lower Bound", lw=lw, alpha=alpha, color=color_lb)
        ax.plot(times, ub, label="Upper Bound", lw=lw, alpha=alpha, color=color_ub)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if log_scale:
            ax.set_yscale("log")

        ax.legend(loc=legend_loc, fontsize=legend_fontsize)

        if grid:
            ax.grid()

        return ax
