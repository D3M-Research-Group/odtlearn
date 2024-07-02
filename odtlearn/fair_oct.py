import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from odtlearn import ODTL
from odtlearn.constrained_oct import ConstrainedOCT
from odtlearn.flow_oct_ms import FlowOCTMultipleSink
from odtlearn.utils.validation import check_binary, check_columns_match


class FairConstrainedOCT(ConstrainedOCT):
    """
    Base class for fair constrained optimal classification trees.

    This class extends the :mod:`ConstrainedOCT <odtlearn.constrained_oct.ConstrainedOCT>` class
    and provides a framework for implementing
    fair constrained optimal classification trees. It includes methods for adding fairness
    constraints, extracting metadata from the input data, and defining the objective function.

    Parameters
    ----------
    solver : str
        The name of the solver to use for solving the MIP problem.
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    _lambda : float
        The regularization parameter in the objective. Must be in the interval [0, 1).
    obj_mode : {'acc', 'balance', 'custom'}, optional (default='acc')
        The objective mode to use.
        'acc' for accuracy, 'balance' for balanced accuracy, 'custom' for user-defined weights.
    fairness_bound: float (0,1], default=1
        The bound of the fairness constraint. The smaller the value the stricter
        the fairness constraint and 1 corresponds to no fairness constraint enforced
    depth : int
        The maximum depth of the tree.
    time_limit : int
        The time limit (in seconds) for solving the MIP problem.
    num_threads : int, optional
        The number of threads to use for solving the MIP problem. If None, all available threads are used.
    verbose : bool, optional
        Whether to display verbose output during the solving process.

    Attributes
    ----------
    _obj_mode : str
        The objective mode used for learning the optimal tree. Must be either 'acc' or 'balance'.
    _positive_class : int
        The value of the positive class label.
    _fairness_bound : float
        The bound of the fairness constraint. Must be in the interval (0, 1].
    _protect_feat_col_labels : list of str
        The column labels of the protected features.
    _protect_feat_col_dtypes : list of dtype
        The data types of the protected feature columns.

    Methods
    -------
    _add_fairness_constraint(p_df, p_prime_df)
        Add the fairness constraint to the MIP problem for the given protected groups.
    _extract_metadata(X, y, protect_feat)
        Extract metadata from the input data.
    _define_objective()
        Define the objective function for the MIP problem.
    fit(X, y, protect_feat, legit_factor)
        Fit the fair constrained optimal classification tree on the given data.
    predict(X)
        Predict the class labels for the given input data using the fitted model.

    Notes
    -----
    This is a base class and should not be instantiated directly. Instead, use one of the
    derived classes that implement a specific fairness constraint, such as
    :mod:`FairSPOCT <odtlearn.fair_oct.FairSPOCT>`,
    :mod:`FairCSPOCT <odtlearn.fair_oct.FairCSPOCT>`,
    :mod:`FairPEOCT <odtlearn.fair_oct.FairPEOCT>`,
    :mod:`FairEOppOCT <odtlearn.fair_oct.FairEOppOCT>`,
    or :mod:`FairEOddsOCT <odtlearn.fair_oct.FairEOddsOCT>`.

    The :meth:`fit <odtlearn.fair_oct.FairConstrainedOCT.fit>` method expects the input data `X`,
    target labels `y`, protected features
    `protect_feat`, and legitimate factors `legit_factor` (if applicable) to be provided.
    The protected features should be binary-valued, and the legitimate factors should be
    numeric.

    The :meth:`predict <odtlearn.fair_oct.FairConstrainedOCT.predict>` method expects the input data
    `X` to have the same columns as the data
    used for fitting the model.

    """

    def __init__(
        self,
        solver,
        positive_class,
        _lambda,
        obj_mode,
        fairness_bound,
        depth,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        self._positive_class = positive_class
        self._fairness_bound = fairness_bound
        self.obj_mode = obj_mode
        if obj_mode not in ["acc", "balance", "custom"]:
            raise ValueError("objective must be one of 'acc', 'balance', or 'custom'")
        self._obj_mode = obj_mode
        self.weights = None
        super().__init__(solver, _lambda, depth, time_limit, num_threads, verbose)

    def _extract_metadata(self, X, y, protect_feat):
        super(ConstrainedOCT, self)._extract_metadata(X, y)
        if isinstance(protect_feat, pd.DataFrame):
            self._protect_feat_col_labels = protect_feat.columns
            self._protect_feat_col_dtypes = protect_feat.dtypes
        else:
            self._protect_feat_col_labels = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

    def _add_fairness_constraint(self, p_df, p_prime_df):
        """
        Add the fairness constraint to the MIP problem.

        Parameters
        ----------
        p_df : pandas.DataFrame
            The subset of the data corresponding to the protected group p.
        p_prime_df : pandas.DataFrame
            The subset of the data corresponding to the protected group p'.

        Returns
        -------
        constraint_added : bool
            Whether the fairness constraint was successfully added.
        """
        count_p = p_df.shape[0]
        count_p_prime = p_prime_df.shape[0]
        constraint_added = False
        if count_p != 0 and count_p_prime != 0:
            constraint_added = True
            self._solver.add_constr(
                (
                    (1 / count_p)
                    * self._solver.quicksum(
                        self._solver.quicksum(
                            self._zeta[i, n, self._positive_class]
                            for n in self._tree.Leaves + self._tree.Nodes
                        )
                        for i in p_df.index
                    )
                    - (
                        (1 / count_p_prime)
                        * self._solver.quicksum(
                            self._solver.quicksum(
                                self._zeta[i, n, self._positive_class]
                                for n in self._tree.Leaves + self._tree.Nodes
                            )
                            for i in p_prime_df.index
                        )
                    )
                )
                <= self._fairness_bound
            )

            self._solver.add_constr(
                (
                    (1 / count_p)
                    * self._solver.quicksum(
                        self._solver.quicksum(
                            self._zeta[i, n, self._positive_class]
                            for n in (self._tree.Leaves + self._tree.Nodes)
                        )
                        for i in p_df.index
                    )
                )
                - (
                    (1 / count_p_prime)
                    * self._solver.quicksum(
                        self._solver.quicksum(
                            self._zeta[i, n, self._positive_class]
                            for n in self._tree.Leaves + self._tree.Nodes
                        )
                        for i in p_prime_df.index
                    )
                )
                >= -1 * self._fairness_bound
            )

        return constraint_added

    def _define_objective(self):
        # Max sum(sum(zeta[i,n,y(i)]))
        obj = self._solver.lin_expr(0)
        for n in self._tree.Nodes:
            for f in self._X_col_labels:
                obj += -1 * self._lambda * self._b[n, f]

        for i in self._datapoints:
            for n in self._tree.Nodes + self._tree.Leaves:
                obj += (
                    (1 - self._lambda)
                    * self.weights[i]
                    * (self._zeta[i, n, self._y[i]])
                )

        self._solver.set_objective(obj, ODTL.MAXIMIZE)

    def fit(self, X, y, protect_feat, legit_factor, weights=None):
        """
        Fit the Fair Constrained Optimal Classification Tree (FairConstrainedOCT) model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each feature should be binary (0 or 1).
        y : array-like of shape (n_samples,)
            The target values (class labels) for the training samples.
        protect_feat : array-like of shape (n_samples, n_protected_features)
            The protected feature columns (e.g., race, gender). Can have one or more columns.
            Each protected feature should be binary (0 or 1).
        legit_factor : array-like of shape (n_samples,)
            The legitimate factor column (e.g., prior number of criminal acts).
            This should be a numeric column.
        weights : array-like of shape (n_samples,), optional (default=None)
            Sample weights. If None, then samples are equally weighted when obj_mode is 'acc',
            or weights are automatically calculated when obj_mode is 'balance'.
            Must be provided when obj_mode is 'custom'.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If X or protect_feat contains non-binary values, or if inputs have inconsistent numbers of samples.
            Also raised if weights are not provided when obj_mode is 'custom', or if the number
            of weights doesn't match the number of samples.
        AssertionError
            If the fairness bound is not in the range (0, 1].

        Notes
        -----
        This method fits the FairConstrainedOCT model using mixed-integer optimization while
        considering fairness constraints. It sets up the optimization problem, solves it, and stores the results.

        The fairness constraints are applied based on the specific fairness metric defined in the subclass
        (e.g., Statistical Parity, Conditional Statistical Parity, Predictive Equality, or Equal Opportunity).

        The optimization problem aims to maximize accuracy (or balanced accuracy, depending on the obj_mode)
        while satisfying the fairness constraints within the specified fairness_bound.

        The resulting tree structure is stored in the model and can be used for prediction or visualization.

        The behavior of this method depends on the `obj_mode` specified during initialization:
        - If obj_mode is 'acc', equal weights are used (weights parameter is ignored).
        - If obj_mode is 'balance', weights are automatically calculated to balance class importance.
        - If obj_mode is 'custom', the provided weights are used.

        When obj_mode is not 'custom' and weights are provided, a warning is issued and the weights are ignored.

        Examples
        --------
        >>> from odtlearn.fair_oct import FairConstrainedOCT
        >>> import numpy as np
        >>> X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        >>> y = np.array([0, 1, 1, 0])
        >>> protect_feat = np.array([[1], [0], [1], [0]])
        >>> legit_factor = np.array([0.1, 0.2, 0.3, 0.4])
        >>> model = FairConstrainedOCT(solver="cbc", positive_class=1, depth=2, fairness_bound=0.1)
        >>> model.fit(X, y, protect_feat, legit_factor)
        """
        self._extract_metadata(X, y, protect_feat)

        self._protect_feat = protect_feat
        self._legit_factor = legit_factor

        self._class_name = "class_label"
        self._legitimate_name = "legitimate_feature_name"

        # this function returns converted X and y but we retain metadata
        if isinstance(y, (pd.Series, pd.DataFrame)):
            X, y = check_X_y(X, y.values.ravel())
        else:
            X, y = check_X_y(X, y)
        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)

        self._X_p = np.concatenate(
            (protect_feat, legit_factor.reshape(-1, 1), y.reshape(-1, 1)), axis=1
        )
        self._X_p = pd.DataFrame(
            self._X_p,
            columns=(
                self._protect_feat_col_labels.tolist()
                + [self._legitimate_name, self._class_name]
            ),
        )

        self._P_col_labels = self._protect_feat_col_labels

        # Store the classes seen during fit
        self._classes = unique_labels(y)

        if weights is not None:
            if self._obj_mode != "custom":
                warnings.warn("Weights are ignored because obj_mode is not 'custom'.")
            elif len(weights) != len(y):
                raise ValueError(
                    "The number of weights must match the number of samples."
                )
            else:
                self.weights = np.array(weights)

        # Generate weights based on obj_mode
        if self._obj_mode == "acc":
            self.weights = np.ones(len(y))
        elif self._obj_mode == "balance":
            class_counts = np.bincount(y)
            self.weights = np.array(
                [1 / (class_counts[yi] * len(self._labels)) for yi in y]
            )
        elif self._obj_mode == "custom":
            if self.weights is None:
                raise ValueError("Weights must be provided when obj_mode is 'custom'.")

        self._create_main_problem()
        self._solver.optimize(self._X, self, self._solver)

        self.b_value = self._solver.get_var_value(self._b, "b")
        self.w_value = self._solver.get_var_value(self._w, "w")
        self.p_value = self._solver.get_var_value(self._p, "p")

        # Return the classifier
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X using the fitted Fair Constrained Optimal Classification Tree model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to make predictions. Each feature should be binary (0 or 1).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for each sample in X.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If X contains non-binary values or has a different number of features than the training data.

        Notes
        -----
        This method uses the fair decision tree learned during the fit process to classify new samples.
        It traverses the tree for each sample in X, following the branching decisions until
        reaching a leaf node, and returns the corresponding class prediction.

        The predictions made by this method satisfy the fairness constraints that were imposed
        during the training process. However, note that the fairness guarantees only hold for the
        distribution of the training data. When applying the model to new data with a different
        distribution, the fairness properties may not be preserved.

        Examples
        --------
        >>> from odtlearn.fair_oct import FairSPOCT
        >>> import numpy as np
        >>> X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        >>> y_train = np.array([0, 1, 1, 0])
        >>> protect_feat = np.array([0, 1, 1, 0])
        >>> legit_factor = np.array([0, 1, 0, 1])
        >>> clf = FairSPOCT(solver="cbc", positive_class=1, depth=2, fairness_bound=0.1)
        >>> clf.fit(X_train, y_train, protect_feat, legit_factor)
        >>> X_test = np.array([[1, 1], [0, 0]])
        >>> y_pred = clf.predict(X_test)
        >>> print(y_pred)
        [1 0]
        """

        # Check is fit had been called
        check_is_fitted(self, ["b_value", "w_value", "p_value"])

        check_columns_match(self._X_col_labels, X)

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        return self._make_prediction(X)


class FairSPOCT(FairConstrainedOCT):
    """
    An optimal classification tree fit on a given binary-valued data set
    with a fairness side-constraint requiring statistical parity (SP) between protected groups.

    Parameters
    ----------
    solver: str
        A string specifying the name of the solver to use
        to solve the MIP. Options are "Gurobi" and "CBC".
        If the CBC binaries are not found, Gurobi will be used by default.
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    depth : int, default = 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default = 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    obj_mode : {'acc', 'balance', 'custom'}, optional (default='acc')
        The objective mode to use.
        'acc' for accuracy, 'balance' for balanced accuracy, 'custom' for user-defined weights.
    fairness_bound: float (0,1], default=1
        The bound of the fairness constraint. The smaller the value the stricter
        the fairness constraint and 1 corresponds to no fairness constraint enforced
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads

    """

    def __init__(
        self,
        solver,
        positive_class,
        depth=1,
        time_limit=60,
        _lambda=0,
        obj_mode="acc",
        fairness_bound=1,
        num_threads=None,
        verbose=False,
    ) -> None:

        super().__init__(
            solver,
            positive_class,
            _lambda,
            obj_mode,
            fairness_bound,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_side_constraints(self):
        # Loop through all possible combinations of the protected feature
        for protected_feature in self._P_col_labels:
            for combo in combinations(self._X_p[protected_feature].unique(), 2):
                p = combo[0]
                p_prime = combo[1]

                p_df = self._X_p[self._X_p[protected_feature] == p]
                p_prime_df = self._X_p[self._X_p[protected_feature] == p_prime]
                self._add_fairness_constraint(p_df, p_prime_df)

    def calc_metric(self, protect_feat, y):
        """
        Calculate the statistical parity metric for the given data.

        Parameters
        ----------
        protect_feat : array-like of shape (n_samples, n_protected_features)
            The protected feature columns (e.g., race, gender). Can have one or more columns.
        y : array-like of shape (n_samples,)
            The target values or predicted values.

        Returns
        -------
        sp_dict : dict
            A dictionary with key (p,t) and value P(Y=t|P=p), where p is a protected level and t is an outcome value.

        Notes
        -----
        This method calculates the statistical parity metric, which measures the difference in prediction rates
        across different protected groups.
        """
        if isinstance(protect_feat, pd.DataFrame):
            protect_feat_test_col_names = protect_feat.columns
        else:
            protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        protect_feat, y = check_X_y(protect_feat, y)

        check_columns_match(self._protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        X_p = np.concatenate((protect_feat, y.reshape(-1, 1)), axis=1)
        X_p = pd.DataFrame(
            X_p,
            columns=(protect_feat_test_col_names.tolist() + [class_name]),
        )

        sp_dict = {}

        for t in X_p[class_name].unique():
            for protected_feature in protect_feat_test_col_names:
                for p in X_p[protected_feature].unique():
                    p_df = X_p[X_p[protected_feature] == p]
                    sp_p_t = None
                    if p_df.shape[0] != 0:
                        sp_p_t = p_df[p_df[class_name] == t].shape[0] / p_df.shape[0]
                    sp_dict[(p, t)] = sp_p_t

        return sp_dict


class FairCSPOCT(FairConstrainedOCT):
    """
    An optimal classification tree fit on a given binary-valued data set
    with a fairness side-constraint requiring conditional statistical parity (CSP) between protected groups.

    Parameters
    ----------
    solver: str
        A string specifying the name of the solver to use
        to solve the MIP. Options are "Gurobi" and "CBC".
        If the CBC binaries are not found, Gurobi will be used by default.
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    depth : int, default = 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default = 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    obj_mode : {'acc', 'balance', 'custom'}, optional (default='acc')
        The objective mode to use.
        'acc' for accuracy, 'balance' for balanced accuracy, 'custom' for user-defined weights.
    fairness_bound: float (0,1], default=1
        The bound of the fairness constraint. The smaller the value the stricter
        the fairness constraint and 1 corresponds to no fairness constraint enforced
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads
    """

    def __init__(
        self,
        solver,
        positive_class,
        depth=1,
        time_limit=60,
        _lambda=0,
        obj_mode="acc",
        fairness_bound=1,
        num_threads=None,
        verbose=False,
    ) -> None:

        super().__init__(
            solver,
            positive_class,
            _lambda,
            obj_mode,
            fairness_bound,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_side_constraints(self):
        # Loop through all possible combinations of the protected feature
        for protected_feature in self._P_col_labels:
            for combo in combinations(self._X_p[protected_feature].unique(), 2):
                p = combo[0]
                p_prime = combo[1]
                for l_value in self._X_p[self._legitimate_name].unique():
                    p_df = self._X_p[
                        (self._X_p[protected_feature] == p)
                        & (self._X_p[self._legitimate_name] == l_value)
                    ]
                    p_prime_df = self._X_p[
                        (self._X_p[protected_feature] == p_prime)
                        & (self._X_p[self._legitimate_name] == l_value)
                    ]
                    self._add_fairness_constraint(p_df, p_prime_df)

    def calc_metric(self, protect_feat, legit_factor, y):
        """
        Calculate the conditional statistical parity metric for the given data.

        Parameters
        ----------
        protect_feat : array-like of shape (n_samples, n_protected_features)
            The protected feature columns (e.g., race, gender). Can have one or more columns.
        legit_factor : array-like of shape (n_samples,)
            The legitimate factor column (e.g., prior number of criminal acts).
        y : array-like of shape (n_samples,)
            The target values or predicted values.

        Returns
        -------
        csp_dict : dict
            A dictionary with key (p, f, t) and value P(Y=t|P=p, L=f), where p is a protected level,
            t is an outcome value, and f is the value of the legitimate feature.

        Notes
        -----
        This method calculates the conditional statistical parity metric, which measures the difference in
        prediction rates across different protected groups, conditioned on the legitimate factor.
        """

        if isinstance(protect_feat, pd.DataFrame):
            protect_feat_test_col_names = protect_feat.columns
        else:
            protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        _, y = check_X_y(protect_feat, y)
        protect_feat, legit_factor = check_X_y(protect_feat, legit_factor)

        check_columns_match(self._protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        legitimate_name = "legitimate_feature_name"
        X_p = np.concatenate(
            (protect_feat, legit_factor.reshape(-1, 1), y.reshape(-1, 1)), axis=1
        )
        X_p = pd.DataFrame(
            X_p,
            columns=(
                protect_feat_test_col_names.tolist() + [legitimate_name, class_name]
            ),
        )

        csp_dict = {}

        for t in X_p[class_name].unique():
            for protected_feature in protect_feat_test_col_names:
                for p in X_p[protected_feature].unique():
                    for f in X_p[legitimate_name].unique():
                        p_f_df = X_p[
                            (X_p[protected_feature] == p) & (X_p[legitimate_name] == f)
                        ]
                        csp_p_f_t = None
                        if p_f_df.shape[0] != 0:
                            csp_p_f_t = (
                                p_f_df[p_f_df[class_name] == t].shape[0]
                            ) / p_f_df.shape[0]
                        csp_dict[(p, f, t)] = csp_p_f_t

        return csp_dict


class FairPEOCT(FairConstrainedOCT):
    """
    An optimal classification tree fit on a given binary-valued data set
    with a fairness side-constraint requiring predictive equity (PE) between protected groups.

            Parameters
    ----------
    solver: str
        A string specifying the name of the solver to use
        to solve the MIP. Options are "Gurobi" and "CBC".
        If the CBC binaries are not found, Gurobi will be used by default.
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    depth : int, default = 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default = 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    obj_mode: str, default="acc"
        The objective should be used to learn an optimal decision tree.
        The two options are "acc" and "balance".
        The accuracy objective attempts to maximize prediction accuracy while the
        balance objective aims to learn a balanced optimal decision
        tree to better generalize to our of sample data.
    fairness_bound: float (0,1], default=1
        The bound of the fairness constraint. The smaller the value the stricter
        the fairness constraint and 1 corresponds to no fairness constraint enforced
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads
    """

    def __init__(
        self,
        solver,
        positive_class,
        depth=1,
        time_limit=60,
        _lambda=0,
        obj_mode="acc",
        fairness_bound=1,
        num_threads=None,
        verbose=False,
    ) -> None:

        super().__init__(
            solver,
            positive_class,
            _lambda,
            obj_mode,
            fairness_bound,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_side_constraints(self):
        # Loop through all possible combinations of the protected feature
        for protected_feature in self._P_col_labels:
            for combo in combinations(self._X_p[protected_feature].unique(), 2):
                p = combo[0]
                p_prime = combo[1]
                p_df = self._X_p[
                    (self._X_p[protected_feature] == p)
                    & (self._X_p[self._class_name] != self._positive_class)
                ]
                p_prime_df = self._X_p[
                    (self._X_p[protected_feature] == p_prime)
                    & (self._X_p[self._class_name] != self._positive_class)
                ]
                self._add_fairness_constraint(p_df, p_prime_df)

    def calc_metric(self, protect_feat, y, y_pred):
        """
        Calculate the predictive equality metric for the given data.

        Parameters
        ----------
        protect_feat : array-like of shape (n_samples, n_protected_features)
            The protected feature columns (e.g., race, gender). Can have one or more columns.
        y : array-like of shape (n_samples,)
            The true target values.
        y_pred : array-like of shape (n_samples,)
            The predicted values.

        Returns
        -------
        eq_dict : dict
            A dictionary with key (p, t, t_pred) and value P(Y_pred=t_pred|P=p, Y=t), where p is a protected level,
            t is a true outcome value, and t_pred is a predicted outcome value.

        Notes
        -----
        This method calculates the predictive equality metric, which measures the difference in
        false positive rates across different protected groups.
        """

        if isinstance(protect_feat, pd.DataFrame):
            protect_feat_test_col_names = protect_feat.columns
        else:
            protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        if isinstance(y, (pd.Series, pd.DataFrame)):
            _, y = check_X_y(protect_feat, y.values.ravel())
        else:
            _, y = check_X_y(protect_feat, y)
        protect_feat, y_pred = check_X_y(protect_feat, y_pred)

        check_columns_match(self._protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        pred_name = "pred_label"
        X_p = np.concatenate(
            (protect_feat, y.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1
        )
        X_p = pd.DataFrame(
            X_p,
            columns=(protect_feat_test_col_names.tolist() + [class_name, pred_name]),
        )

        eq_dict = {}

        for t in X_p[class_name].unique():
            for t_pred in X_p[class_name].unique():
                for protected_feature in protect_feat_test_col_names:
                    for p in X_p[protected_feature].unique():
                        p_t_df = X_p[
                            (X_p[protected_feature] == p) & (X_p[class_name] == t)
                        ]
                        eq_p_t_t_pred = None
                        if p_t_df.shape[0] != 0:
                            eq_p_t_t_pred = (
                                p_t_df[p_t_df[pred_name] == t_pred].shape[0]
                            ) / p_t_df.shape[0]
                        eq_dict[(p, t, t_pred)] = eq_p_t_t_pred

        return eq_dict


class FairEOppOCT(FairConstrainedOCT):
    """
    An optimal classification tree fit on a given binary-valued data set
    with a fairness side-constraint requiring equality of opportunity (EOpp) between protected groups.

    Parameters
    ----------
    solver: str
        A string specifying the name of the solver to use
        to solve the MIP. Options are "Gurobi" and "CBC".
        If the CBC binaries are not found, Gurobi will be used by default.
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    depth : int, default = 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default = 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    obj_mode: str, default="acc"
        The objective should be used to learn an optimal decision tree.
        The two options are "acc" and "balance".
        The accuracy objective attempts to maximize prediction accuracy while the
        balance objective aims to learn a balanced optimal decision
        tree to better generalize to our of sample data.
    fairness_bound: float (0,1], default=1
        The bound of the fairness constraint. The smaller the value the stricter
        the fairness constraint and 1 corresponds to no fairness constraint enforced
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads
    """

    def __init__(
        self,
        solver,
        positive_class,
        depth=1,
        time_limit=60,
        _lambda=0,
        obj_mode="acc",
        fairness_bound=1,
        num_threads=None,
        verbose=False,
    ) -> None:

        super().__init__(
            solver,
            positive_class,
            _lambda,
            obj_mode,
            fairness_bound,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_side_constraints(self):
        # Loop through all possible combinations of the protected feature
        for protected_feature in self._P_col_labels:
            for combo in combinations(self._X_p[protected_feature].unique(), 2):
                p = combo[0]
                p_prime = combo[1]
                p_df = self._X_p[
                    (self._X_p[protected_feature] == p)
                    & (self._X_p[self._class_name] == self._positive_class)
                ]
                p_prime_df = self._X_p[
                    (self._X_p[protected_feature] == p_prime)
                    & (self._X_p[self._class_name] == self._positive_class)
                ]
                self._add_fairness_constraint(p_df, p_prime_df)

    def calc_metric(self):
        raise NotImplementedError()


class FairEOddsOCT(FairConstrainedOCT):
    """
    An optimal classification tree fit on a given binary-valued data set
    with a fairness side-constraint requiring equal oddts (EOdds) between protected groups.

            Parameters
    ----------
    solver: str
        A string specifying the name of the solver to use
        to solve the MIP. Options are "Gurobi" and "CBC".
        If the CBC binaries are not found, Gurobi will be used by default.
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    depth : int, default = 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default = 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    obj_mode: str, default="acc"
        The objective should be used to learn an optimal decision tree.
        The two options are "acc" and "balance".
        The accuracy objective attempts to maximize prediction accuracy while the
        balance objective aims to learn a balanced optimal decision
        tree to better generalize to our of sample data.
    fairness_bound: float (0,1], default=1
        The bound of the fairness constraint. The smaller the value the stricter
        the fairness constraint and 1 corresponds to no fairness constraint enforced
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads
    """

    def __init__(
        self,
        solver,
        positive_class,
        depth=1,
        time_limit=60,
        _lambda=0,
        obj_mode="acc",
        fairness_bound=1,
        num_threads=None,
        verbose=False,
    ) -> None:

        super().__init__(
            solver,
            positive_class,
            _lambda,
            obj_mode,
            fairness_bound,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_side_constraints(self):
        # Loop through all possible combinations of the protected feature
        for protected_feature in self._P_col_labels:
            for combo in combinations(self._X_p[protected_feature].unique(), 2):
                p = combo[0]
                p_prime = combo[1]
                # TODO: Need to check with group if this is how we want to enforce this constraint
                PE_p_df = self._X_p[
                    (self._X_p[protected_feature] == p)
                    & (self._X_p[self._class_name] != self._positive_class)
                ]
                PE_p_prime_df = self._X_p[
                    (self._X_p[protected_feature] == p_prime)
                    & (self._X_p[self._class_name] != self._positive_class)
                ]

                EOpp_p_df = self._X_p[
                    (self._X_p[protected_feature] == p)
                    & (self._X_p[self._class_name] == self._positive_class)
                ]
                EOpp_p_prime_df = self._X_p[
                    (self._X_p[protected_feature] == p_prime)
                    & (self._X_p[self._class_name] == self._positive_class)
                ]

                if (
                    PE_p_df.shape[0] != 0
                    and PE_p_prime_df.shape[0] != 0
                    and EOpp_p_df.shape[0] != 0
                    and EOpp_p_prime_df.shape[0] != 0
                ):
                    self._add_fairness_constraint(PE_p_df, PE_p_prime_df)
                    self._add_fairness_constraint(EOpp_p_df, EOpp_p_prime_df)


class FairOCT(FlowOCTMultipleSink):
    """
    An optimal and fair classification tree fitted on a given binary-valued
    data set. The fairness criteria enforced in the training step is one of statistical parity (SP),
    conditional statistical parity (CSP), predictive equality (PE),
    equal opportunity (EOpp) or equalized odds (EOdds).

    Parameters
    ----------
    solver: str
        A string specifying the name of the solver to use
        to solve the MIP. Options are "Gurobi" and "CBC".
        If the CBC binaries are not found, Gurobi will be used by default.
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    depth : int, default= 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default= 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads
    fairness_type: [None, 'SP', 'CSP', 'PE', 'EOpp', 'EOdds'], default=None
        The type of fairness criteria that we want to enforce
    fairness_bound: float (0,1], default=1
        The bound of the fairness constraint. The smaller the value the stricter
        the fairness constraint and 1 corresponds to no fairness constraint enforced
    """

    def __init__(
        self,
        solver,
        positive_class,
        _lambda=0,
        depth=1,
        obj_mode="acc",
        fairness_type=None,
        fairness_bound=1,
        time_limit=60,
        num_threads=None,
        verbose=False,
    ) -> None:

        warnings.warn(
            (
                "The class FairOCT will be depreciated in v1.2 of ODTlearn,"
                "please use one of the metric-specific classes, e.g., FairSPOCT, FairCSPOCT, FairPEOCT, etc."
            ),
            PendingDeprecationWarning,
        )
        super().__init__(
            solver,
            _lambda,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

        self._obj_mode = obj_mode
        self._fairness_type = fairness_type
        self._fairness_bound = fairness_bound
        self._positive_class = positive_class

    def _extract_metadata(self, X, y, protect_feat):
        """
        Extract metadata from the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        protect_feat : array-like of shape (n_samples, n_protect_feat)
            The protected features.
        """
        super(FlowOCTMultipleSink, self)._extract_metadata(X, y)
        if isinstance(protect_feat, pd.DataFrame):
            self._protect_feat_col_labels = protect_feat.columns
            self._protect_feat_col_dtypes = protect_feat.dtypes
        else:
            self._protect_feat_col_labels = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

    def _add_fairness_constraint(self, p_df, p_prime_df):
        count_p = p_df.shape[0]
        count_p_prime = p_prime_df.shape[0]
        constraint_added = False
        if count_p != 0 and count_p_prime != 0:
            constraint_added = True
            self._solver.add_constr(
                (
                    (1 / count_p)
                    * self._solver.quicksum(
                        self._solver.quicksum(
                            self._zeta[i, n, self._positive_class]
                            for n in self._tree.Leaves + self._tree.Nodes
                        )
                        for i in p_df.index
                    )
                    - (
                        (1 / count_p_prime)
                        * self._solver.quicksum(
                            self._solver.quicksum(
                                self._zeta[i, n, self._positive_class]
                                for n in self._tree.Leaves + self._tree.Nodes
                            )
                            for i in p_prime_df.index
                        )
                    )
                )
                <= self._fairness_bound
            )

            self._solver.add_constr(
                (
                    (1 / count_p)
                    * self._solver.quicksum(
                        self._solver.quicksum(
                            self._zeta[i, n, self._positive_class]
                            for n in (self._tree.Leaves + self._tree.Nodes)
                        )
                        for i in p_df.index
                    )
                )
                - (
                    (1 / count_p_prime)
                    * self._solver.quicksum(
                        self._solver.quicksum(
                            self._zeta[i, n, self._positive_class]
                            for n in self._tree.Leaves + self._tree.Nodes
                        )
                        for i in p_prime_df.index
                    )
                )
                >= -1 * self._fairness_bound
            )

        return constraint_added

    def _define_constraints(self):
        super()._define_constraints()
        # Loop through all possible combinations of the protected feature
        for protected_feature in self._P_col_labels:
            for combo in combinations(self._X_p[protected_feature].unique(), 2):
                p = combo[0]
                p_prime = combo[1]

                if self._fairness_type == "SP":
                    p_df = self._X_p[self._X_p[protected_feature] == p]
                    p_prime_df = self._X_p[self._X_p[protected_feature] == p_prime]
                    self._add_fairness_constraint(p_df, p_prime_df)
                elif self._fairness_type == "PE":
                    p_df = self._X_p[
                        (self._X_p[protected_feature] == p)
                        & (self._X_p[self._class_name] != self._positive_class)
                    ]
                    p_prime_df = self._X_p[
                        (self._X_p[protected_feature] == p_prime)
                        & (self._X_p[self._class_name] != self._positive_class)
                    ]
                    self._add_fairness_constraint(p_df, p_prime_df)
                elif self._fairness_type == "EOpp":
                    p_df = self._X_p[
                        (self._X_p[protected_feature] == p)
                        & (self._X_p[self._class_name] == self._positive_class)
                    ]
                    p_prime_df = self._X_p[
                        (self._X_p[protected_feature] == p_prime)
                        & (self._X_p[self._class_name] == self._positive_class)
                    ]
                    self._add_fairness_constraint(p_df, p_prime_df)
                elif (
                    self._fairness_type == "EOdds"
                ):  # Need to check with group if this is how we want to enforce this constraint
                    PE_p_df = self._X_p[
                        (self._X_p[protected_feature] == p)
                        & (self._X_p[self._class_name] != self._positive_class)
                    ]
                    PE_p_prime_df = self._X_p[
                        (self._X_p[protected_feature] == p_prime)
                        & (self._X_p[self._class_name] != self._positive_class)
                    ]

                    EOpp_p_df = self._X_p[
                        (self._X_p[protected_feature] == p)
                        & (self._X_p[self._class_name] == self._positive_class)
                    ]
                    EOpp_p_prime_df = self._X_p[
                        (self._X_p[protected_feature] == p_prime)
                        & (self._X_p[self._class_name] == self._positive_class)
                    ]

                    if (
                        PE_p_df.shape[0] != 0
                        and PE_p_prime_df.shape[0] != 0
                        and EOpp_p_df.shape[0] != 0
                        and EOpp_p_prime_df.shape[0] != 0
                    ):
                        self._add_fairness_constraint(PE_p_df, PE_p_prime_df)
                        self._add_fairness_constraint(EOpp_p_df, EOpp_p_prime_df)
                elif self._fairness_type == "CSP":
                    for l_value in self._X_p[self._legitimate_name].unique():
                        p_df = self._X_p[
                            (self._X_p[protected_feature] == p)
                            & (self._X_p[self._legitimate_name] == l_value)
                        ]
                        p_prime_df = self._X_p[
                            (self._X_p[protected_feature] == p_prime)
                            & (self._X_p[self._legitimate_name] == l_value)
                        ]
                        self._add_fairness_constraint(p_df, p_prime_df)

    def _define_objective(self):
        # Max sum(sum(zeta[i,n,y(i)]))
        obj = self._solver.lin_expr(0)
        for n in self._tree.Nodes:
            for f in self._X_col_labels:
                obj += -1 * self._lambda * self._b[n, f]
        if self._obj_mode == "acc":
            for i in self._datapoints:
                for n in self._tree.Nodes + self._tree.Leaves:
                    obj += (1 - self._lambda) * (self._zeta[i, n, self._y[i]])
        elif self._obj_mode == "balance":
            for i in self._datapoints:
                for n in self._tree.Nodes + self._tree.Leaves:
                    obj += (
                        (1 - self._lambda)
                        * (
                            1
                            / self._y[self._y == self._y[i]].shape[0]
                            / self._labels.shape[0]
                        )
                        * (self._zeta[i, n, self._y[i]])
                    )
        else:
            raise ValueError(
                "Invalid objective mode. obj_mode should be one of acc or balance."
            )
        self._solver.set_objective(obj, ODTL.MAXIMIZE)

    def fit(self, X, y, protect_feat, legit_factor):
        """
        Fit the FairOCT model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each feature should be binary (0 or 1).
        y : array-like of shape (n_samples,)
            The target values (class labels) for the training samples.
        protect_feat : array-like of shape (n_samples, n_protected_features)
            The protected feature columns (e.g., race, gender). Can have one or more columns.
        legit_factor : array-like of shape (n_samples,)
            The legitimate factor column (e.g., prior number of criminal acts).

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If X contains non-binary values or if inputs have inconsistent numbers of samples.

        Notes
        -----
        This method fits the FairOCT model using mixed-integer optimization while
        considering fairness constraints. It sets up the optimization problem,
        solves it, and stores the results.
        """
        self._extract_metadata(X, y, protect_feat)
        self._protect_feat = protect_feat
        self._legit_factor = legit_factor
        self._class_name = "class_label"
        self._legitimate_name = "legitimate_feature_name"
        # this function returns converted X and y but we retain metadata
        if isinstance(y, (pd.Series, pd.DataFrame)):
            X, y = check_X_y(X, y.values.ravel())
        else:
            X, y = check_X_y(X, y)
        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)
        self._X_p = np.concatenate(
            (protect_feat, legit_factor.reshape(-1, 1), y.reshape(-1, 1)), axis=1
        )
        self._X_p = pd.DataFrame(
            self._X_p,
            columns=(
                self._protect_feat_col_labels.tolist()
                + [self._legitimate_name, self._class_name]
            ),
        )
        self._P_col_labels = self._protect_feat_col_labels
        # Store the classes seen during fit
        self._classes = unique_labels(y)
        self._create_main_problem()
        self._solver.optimize(self._X, self, self._solver)
        self.b_value = self._solver.get_var_value(self._b, "b")
        self.w_value = self._solver.get_var_value(self._w, "w")
        self.p_value = self._solver.get_var_value(self._p, "p")
        # Return the classifier
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X using the fitted FairOCT model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to make predictions. Each feature should be binary (0 or 1).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for each sample in X.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If X contains non-binary values or has a different number of features than the training data.

        Notes
        -----
        This method uses the fair decision tree learned during the fit process to classify new samples.
        It traverses the tree for each sample in X, following the branching decisions until
        reaching a leaf node, and returns the corresponding class prediction.
        """
        # Check is fit had been called
        check_is_fitted(self, ["b_value", "w_value", "p_value"])
        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)
        check_columns_match(self._X_col_labels, X)
        return self._make_prediction(X)

    def get_SP(self, protect_feat, y):
        """
        This function returns the statistical parity value for any given protected level and outcome value

        :param protect_feat: array-like, shape (n_samples,1) or (n_samples, n_p)
                The protected feature columns (Race, gender, etc); We could have one or more columns
        :param y: array-like, shape (n_samples,)
                The target values (class labels in classification).

        :return sp_dict: a dictionary with key =(p,t) and value = P(Y=t|P=p)
        where p is a protected level and t is an outcome value

        """
        if isinstance(protect_feat, pd.DataFrame):
            protect_feat_test_col_names = protect_feat.columns
        else:
            protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        protect_feat, y = check_X_y(protect_feat, y)

        check_columns_match(self._protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        X_p = np.concatenate((protect_feat, y.reshape(-1, 1)), axis=1)
        X_p = pd.DataFrame(
            X_p,
            columns=(protect_feat_test_col_names.tolist() + [class_name]),
        )

        sp_dict = {}

        for t in X_p[class_name].unique():
            for protected_feature in protect_feat_test_col_names:
                for p in X_p[protected_feature].unique():
                    p_df = X_p[X_p[protected_feature] == p]
                    sp_p_t = None
                    if p_df.shape[0] != 0:
                        sp_p_t = p_df[p_df[class_name] == t].shape[0] / p_df.shape[0]
                    sp_dict[(p, t)] = sp_p_t

        return sp_dict

    def get_CSP(self, protect_feat, legit_factor, y):
        """
        This function returns the conditional statistical parity value for any given
        protected level, legitimate feature value and outcome value

        :param protect_feat: array-like, shape (n_samples,1) or (n_samples, n_p)
                The protected feature columns (Race, gender, etc); We could have one or more columns
        :param legit_fact: array-like, shape (n_samples,)
            The legitimate factor column(e.g., prior number of criminal acts)
        :param y: array-like, shape (n_samples,)
                The target values (class labels in classification).


        :return csp_dict: a dictionary with key =(p, f, t) and value = P(Y=t|P=p, L=f) where p is a protected level
                          and t is an outcome value and l is the value of the legitimate feature

        """

        if isinstance(protect_feat, pd.DataFrame):
            protect_feat_test_col_names = protect_feat.columns
        else:
            protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        _, y = check_X_y(protect_feat, y)
        protect_feat, legit_factor = check_X_y(protect_feat, legit_factor)

        check_columns_match(self._protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        legitimate_name = "legitimate_feature_name"
        X_p = np.concatenate(
            (protect_feat, legit_factor.reshape(-1, 1), y.reshape(-1, 1)), axis=1
        )
        X_p = pd.DataFrame(
            X_p,
            columns=(
                protect_feat_test_col_names.tolist() + [legitimate_name, class_name]
            ),
        )

        csp_dict = {}

        for t in X_p[class_name].unique():
            for protected_feature in protect_feat_test_col_names:
                for p in X_p[protected_feature].unique():
                    for f in X_p[legitimate_name].unique():
                        p_f_df = X_p[
                            (X_p[protected_feature] == p) & (X_p[legitimate_name] == f)
                        ]
                        csp_p_f_t = None
                        if p_f_df.shape[0] != 0:
                            csp_p_f_t = (
                                p_f_df[p_f_df[class_name] == t].shape[0]
                            ) / p_f_df.shape[0]
                        csp_dict[(p, f, t)] = csp_p_f_t

        return csp_dict

    def get_EqOdds(self, protect_feat, y, y_pred):
        """
        This function returns the false positive and true positive rate value
        for any given protected level, outcome value and prediction value

        :param protect_feat: array-like, shape (n_samples,1) or (n_samples, n_p)
                The protected feature columns (Race, gender, etc); We could have one or more columns

        :param y: array-like, shape (n_samples,)
                The true target values (class labels in classification).
        :param y_pred: array-like, shape (n_samples,)
                The predicted values (class labels in classification).

        :return eq_dict: a dictionary with key =(p, t, t_pred) and value = P(Y_pred=t_pred|P=p, Y=t)

        """

        if isinstance(protect_feat, pd.DataFrame):
            protect_feat_test_col_names = protect_feat.columns
        else:
            protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        if isinstance(y, (pd.Series, pd.DataFrame)):
            _, y = check_X_y(protect_feat, y.values.ravel())
        else:
            _, y = check_X_y(protect_feat, y)
        protect_feat, y_pred = check_X_y(protect_feat, y_pred)

        check_columns_match(self._protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        pred_name = "pred_label"
        X_p = np.concatenate(
            (protect_feat, y.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1
        )
        X_p = pd.DataFrame(
            X_p,
            columns=(protect_feat_test_col_names.tolist() + [class_name, pred_name]),
        )

        eq_dict = {}

        for t in X_p[class_name].unique():
            for t_pred in X_p[class_name].unique():
                for protected_feature in protect_feat_test_col_names:
                    for p in X_p[protected_feature].unique():
                        p_t_df = X_p[
                            (X_p[protected_feature] == p) & (X_p[class_name] == t)
                        ]
                        eq_p_t_t_pred = None
                        if p_t_df.shape[0] != 0:
                            eq_p_t_t_pred = (
                                p_t_df[p_t_df[pred_name] == t_pred].shape[0]
                            ) / p_t_df.shape[0]
                        eq_dict[(p, t, t_pred)] = eq_p_t_t_pred

        return eq_dict

    def get_CondEqOdds(self, protect_feat, legit_factor, y, y_pred):
        """
        This function returns the conditional false negative and true positive rate value
        for any given protected level, outcome value, prediction value and legitimate feature value

        :param protect_feat: array-like, shape (n_samples,1) or (n_samples, n_p)
                The protected feature columns (Race, gender, etc); We could have one or more columns
        :param legit_factor: array-like, shape (n_samples,)
            The legitimate factor column(e.g., prior number of criminal acts)

        :param y: array-like, shape (n_samples,)
                The true target values (class labels in classification).
        :param y_pred: array-like, shape (n_samples,)
                The predicted values (class labels in classification).

        :return ceq_dict: a dictionary with key =(p, f, t, t_pred) and value = P(Y_pred=t_pred|P=p, Y=t, L=f)

        """

        if isinstance(protect_feat, pd.DataFrame):
            protect_feat_test_col_names = protect_feat.columns
        else:
            protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        if isinstance(y, (pd.Series, pd.DataFrame)):
            _, y = check_X_y(protect_feat, y.values.ravel())
        else:
            _, y = check_X_y(protect_feat, y)
        _, y_pred = check_X_y(protect_feat, y_pred)
        protect_feat, legit_factor = check_X_y(protect_feat, legit_factor)

        check_columns_match(self._protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        pred_name = "pred_label"
        legitimate_name = "legitimate_feature_name"
        X_p = np.concatenate(
            (
                protect_feat,
                legit_factor.reshape(-1, 1),
                y.reshape(-1, 1),
                y_pred.reshape(-1, 1),
            ),
            axis=1,
        )
        X_p = pd.DataFrame(
            X_p,
            columns=(
                protect_feat_test_col_names.tolist()
                + [legitimate_name, class_name, pred_name]
            ),
        )

        ceq_dict = {}

        for t in X_p[class_name].unique():
            for t_pred in X_p[class_name].unique():
                for protected_feature in protect_feat_test_col_names:
                    for p in X_p[protected_feature].unique():
                        for f in X_p[legitimate_name].unique():
                            p_f_t_df = X_p[
                                (X_p[protected_feature] == p)
                                & (X_p[legitimate_name] == f)
                                & (X_p[class_name] == t)
                            ]
                            ceq_p_f_t_t_pred = None
                            if p_f_t_df.shape[0] != 0:
                                ceq_p_f_t_t_pred = (
                                    p_f_t_df[p_f_t_df[pred_name] == t_pred].shape[0]
                                ) / p_f_t_df.shape[0]
                            ceq_dict[(p, f, t, t_pred)] = ceq_p_f_t_t_pred

        return ceq_dict

    def fairness_metric_summary(self, metric, new_data=None):
        """
        Summarize the specified fairness metric for the fitted model.

        Parameters
        ----------
        metric : str
            The name of the fairness metric to summarize. Must be one of 'SP', 'CSP', 'PE', or 'CPE'.
        new_data : array-like of shape (n_samples,), optional
            The new predicted data to use for calculating the fairness metric. If None, the predict method
            is called on the training data to obtain the predicted values. Default is None.

        Raises
        ------
        ValueError
            If the specified metric is not one of the supported options.

        Returns
        -------
        None
            The method prints the fairness metric summary as a pandas DataFrame.

        Notes
        -----
        This method summarizes the specified fairness metric for the fitted model. The supported fairness metrics are:
        - 'SP': Statistical Parity
        - 'CSP': Conditional Statistical Parity
        - 'PE': Predictive Equality
        - 'CPE': Conditional Predictive Equality

        The method checks if the model has been fitted and raises an error if not. If `new_data` is not provided,
        the predict method is called on the training data to obtain the predicted values.

        The fairness metric summary is printed as a pandas DataFrame, showing the metric values for each
        combination of protected attribute, legitimate factor (if applicable), true label, and predicted label
        (if applicable), depending on the selected metric.

        Examples
        --------
        >>> model.fit(X_train, y_train, protect_feat_train, legit_factor_train)
        >>> model.fairness_metric_summary('SP')
                    (p,y)  P(Y=y|P=p)
        0     (Male, False)    0.752475
        1      (Male, True)    0.247525
        2   (Female, False)    0.742574
        3    (Female, True)    0.257426
        """
        check_is_fitted(self, ["b_value", "w_value", "p_value"])
        metric_names = ["SP", "CSP", "PE", "CPE"]
        if new_data is None:
            new_data = self.predict(self._X)
        if metric not in metric_names:
            raise ValueError(
                f"metric argument: '{metric}' does not match any of the options: {metric_names}"
            )
        if metric == "SP":
            sp_df = pd.DataFrame(
                self.get_SP(self._protect_feat, new_data).items(),
                columns=["(p,y)", "P(Y=y|P=p)"],
            )
            print(sp_df)
        elif metric == "CSP":
            csp_df = pd.DataFrame(
                self.get_CSP(self._protect_feat, self._legit_factor, new_data).items(),
                columns=["(p, f, y)", "P(Y=y|P=p, L=f)"],
            )
            print(csp_df)
        elif metric == "PE":
            pe_df = pd.DataFrame(
                self.get_EqOdds(self._protect_feat, self._y, new_data).items(),
                columns=["(p, y, y_pred)", "P(Y_pred=y_pred|P=p, Y=y)"],
            )
            print(pe_df)
        elif metric == "CPE":
            cpe_df = pd.DataFrame(
                self.get_CondEqOdds(
                    self._protect_feat, self._legit_factor, self._y, new_data
                ).items(),
                columns=["(p, f, t, t_pred)", "P(Y_pred=y_pred|P=p, Y=y, L=f)"],
            )
            print(cpe_df)
