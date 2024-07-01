import warnings

import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from odtlearn import ODTL
from odtlearn.flow_oct_ss import FlowOCTSingleSink
from odtlearn.utils.callbacks import BendersCallback
from odtlearn.utils.validation import check_binary, check_columns_match


class FlowOCT(FlowOCTSingleSink):
    """
    An optimal decision tree classifier, fitted on a given integer-valued data set
    to produce a provably optimal decision tree.

    Parameters
    ----------
    solver : str
        The solver to use for the MIP formulation. Currently, only "gurobi" and "CBC" are supported.
    _lambda : float, default=0
        The regularization parameter for controlling the complexity of the learned tree.
    obj_mode : {'acc', 'balance', 'custom'}, optional (default='acc')
        The objective mode to use.
        'acc' for accuracy, 'balance' for balanced accuracy, 'custom' for user-defined weights.
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
    _define_objective()
        Define the objective function for the optimization problem.

    Notes
    -----
    This class extends the :mod:`FlowOCTSingleSink <odtlearn.flow_oct_ss.FlowOCTSingleSink>` class
    and provides the implementation for learning
    optimal classification trees using the flow-based formulation with a single sink node.

    The :mod:`FlowOCT <odtlearn.flow_oct.FlowOCT>` class is a user-facing class that can be instantiated
    directly to learn optimal
    classification trees. It inherits the basic structure and functionality from the
    :mod:`FlowOCTSingleSink <odtlearn.flow_oct_ss.FlowOCTSingleSink>`
    class and adds the specific objective function and model fitting process.

    The class supports two objective modes: "acc" (accuracy) and "balance". The accuracy objective
    aims to maximize the prediction accuracy of the learned tree, while the balance objective aims
    to learn a balanced optimal decision tree to better generalize to out-of-sample data.

    The The :meth:`fit <odtlearn.flow_oct.FlowOCT.fit>` method method is used to fit the optimal
    classification tree to the given training data. It
    preprocesses the input data, creates the main optimization problem, and solves it using the
    specified solver and parameters.

    The :meth:`predict <odtlearn.flow_oct.FlowOCT.predict>` method is used to make predictions using
    the fitted optimal classification tree.
    It takes the input data and traverses the tree based on the learned branching and prediction
    decisions to generate the predictions.

    The :meth:`_define_objective <odtlearn.flow_oct.FlowOCT._define_objective>` method defines the
    objective function for the optimization problem based
    on the selected objective mode. It incorporates the regularization term and the accuracy or
    balance objective term.

    Users can instantiate the :mod:`FlowOCT <odtlearn.flow_oct.FlowOCT>` class directly and use it to
    learn optimal classification
    trees by calling the :meth:`fit <odtlearn.flow_oct.FlowOCT.fit>` method with the training data and
    then using the :meth:`predict <odtlearn.flow_oct.FlowOCT.predict>` method
    to make predictions on new data.
    """

    def __init__(
        self,
        solver,
        _lambda=0,
        obj_mode="acc",
        depth=1,
        time_limit=60,
        num_threads=None,
        verbose=False,
    ) -> None:
        super().__init__(
            solver,
            _lambda,
            depth,
            time_limit,
            num_threads,
            verbose,
        )
        if obj_mode not in ["acc", "balance", "custom"]:
            raise ValueError("objective must be one of 'acc', 'balance', or 'custom'")
        self._obj_mode = obj_mode
        self.weights = None

    def _define_objective(self):
        obj = self._solver.lin_expr(0)
        for n in self._tree.Nodes:
            for f in self._X_col_labels:
                obj += -1 * self._lambda * self._b[n, f]

        for i in self._datapoints:
            obj += (1 - self._lambda) * self.weights[i] * self._z[i, 1]

        self._solver.set_objective(obj, ODTL.MAXIMIZE)

    def fit(self, X, y, weights=None):
        """
        Fit the FlowOCT model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each feature should be binary (0 or 1).
        y : array-like of shape (n_samples,)
            The target values (class labels) for the training samples.
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
            If X contains non-binary values, or if X and y have inconsistent numbers of samples.
            Also raised if weights are not provided when obj_mode is 'custom', or if the number
            of weights doesn't match the number of samples.

        Notes
        -----
        The behavior of this method depends on the `obj_mode` specified during initialization:
        - If obj_mode is 'acc', equal weights are used (weights parameter is ignored).
        - If obj_mode is 'balance', weights are automatically calculated to balance class importance.
        - If obj_mode is 'custom', the provided weights are used.

        When obj_mode is not 'custom' and weights are provided, a warning is issued and the weights are ignored.

        This method fits the FlowOCT model using mixed-integer optimization.
        It sets up the optimization problem, solves it, and stores the results.
        """

        # extract column labels, unique classes and store X as a DataFrame
        self._extract_metadata(X, y)

        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)
        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)

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
        self._solver.optimize(
            self._X, self, self._solver, callback=False, callback_action=None
        )
        self.b_value = self._solver.get_var_value(self._b, "b")
        self.w_value = self._solver.get_var_value(self._w, "w")
        self.p_value = self._solver.get_var_value(self._p, "p")

        return self

    def predict(self, X):
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
        check_is_fitted(self, ["b_value", "w_value", "p_value"])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self._X_col_labels, X)

        return self._make_prediction(X)


class BendersOCT(FlowOCTSingleSink):
    """
    An optimal decision tree classifier using Benders' decomposition, fitted on a given integer-valued
    data set to produce a provably optimal decision tree.

    Parameters
    ----------
    solver : str
        The solver to use for the MIP formulation. Currently, only "gurobi" and "CBC" are supported.
    _lambda : float, default=0
        The regularization parameter for controlling the complexity of the learned tree.
    obj_mode : {'acc', 'balance', 'custom'}, optional (default='acc')
        The objective mode to use.
        'acc' for accuracy, 'balance' for balanced accuracy, 'custom' for user-defined weights.
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
        Fit the optimal classification tree using Benders' decomposition to the given training data.
    predict(X)
        Make predictions using the fitted optimal classification tree.
    _define_variables()
        Define the decision variables used in the Benders' decomposition formulation.
    _define_constraints()
        Define the constraints used in the Benders' decomposition formulation.
    _define_objective()
        Define the objective function for the Benders' decomposition formulation.

    Notes
    -----
    This class extends the :mod:`FlowOCTSingleSink <odtlearn.flow_oct_ss.FlowOCTSingleSink>` class
    and provides the implementation for learning
    optimal classification trees using Benders' decomposition with a single sink node.

    The :mod:`BendersOCT <odtlearn.flow_oct.BendersOCT>` class is a user-facing class that
    can be instantiated directly to learn optimal
    classification trees using Benders' decomposition. It inherits the basic structure and functionality
    from the :mod:`FlowOCTSingleSink <odtlearn.flow_oct_ss.FlowOCTSingleSink>` class and
    adds the specific Benders' decomposition formulation.

    Benders' decomposition is a technique used to solve large-scale optimization problems by decomposing
    them into smaller subproblems. In the context of optimal classification trees, Benders' decomposition
    is used to decompose the problem into a master problem and multiple subproblems, one for each datapoint.

    The master problem focuses on the tree structure and branching decisions, while the subproblems
    optimize the class predictions for each datapoint based on the fixed tree structure. The master
    problem and subproblems are solved iteratively until convergence is reached.

    The :meth:`_define_variables <odtlearn.flow_oct.BendersOCT._define_variables>` method defines
    the decision variables specific to the Benders' decomposition
    formulation, including the variables for the subproblem objective values.

    The :meth:`_define_constraints <odtlearn.flow_oct.BendersOCT._define_constraints>` method defines
    the constraints specific to the Benders' decomposition
    formulation, which ensure the proper linking of the master problem and subproblems.

    The :meth:`_define_objective <odtlearn.flow_oct.BendersOCT._define_objective>` method defines the
    objective function for the Benders' decomposition
    formulation, which includes the regularization term and the objective values from the subproblems.

    The :meth:`fit <odtlearn.flow_oct.BendersOCT.fit>` method is used to fit the optimal classification
    tree using Benders' decomposition to
    the given training data. It preprocesses the input data, creates the main optimization problem,
    and solves it using the specified solver and parameters.

    The :meth:`predict <odtlearn.flow_oct.BendersOCT.predict>` method is used to make predictions using
    the fitted optimal classification tree.
    It takes the input data and traverses the tree based on the learned branching and prediction
    decisions to generate the predictions.

    Users can instantiate the :mod:`BendersOCT <odtlearn.flow_oct.BendersOCT>` class directly and
    use it to learn optimal classification
    trees using Benders' decomposition by calling the :meth:`fit <odtlearn.flow_oct.BendersOCT.fit>`
    method with the training data and then
    using the :meth:`predict <odtlearn.flow_oct.BendersOCT.predict>` method to make predictions on new data.
    """

    def __init__(
        self,
        solver,
        _lambda=0,
        obj_mode="acc",
        depth=1,
        time_limit=60,
        num_threads=None,
        verbose=False,
    ) -> None:

        super().__init__(
            solver,
            _lambda,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

        self._lambda = _lambda
        if obj_mode not in ["acc", "balance", "custom"]:
            raise ValueError("objective must be one of 'acc', 'balance', or 'custom'")
        self._obj_mode = obj_mode
        self.weights = None

    def _define_variables(self):
        self._tree_struc_variables()

        # g[i] is the objective value for the sub-problem[i]
        self._g = self._solver.add_vars(
            self._datapoints, vtype=ODTL.CONTINUOUS, ub=1, name="g"
        )

    def _define_constraints(self):
        self._tree_structure_constraints()

    def _define_objective(self):
        obj = self._solver.lin_expr(0)
        for n in self._tree.Nodes:
            for f in self._X_col_labels:
                obj += -1 * self._lambda * self._b[n, f]

        for i in self._datapoints:
            obj += (1 - self._lambda) * self.weights[i] * self._g[i]

        self._solver.set_objective(obj, ODTL.MAXIMIZE)

    def fit(self, X, y, weights=None):
        """
        Fit the BendersOCT model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each feature should be binary (0 or 1).
        y : array-like of shape (n_samples,)
            The target values (class labels) for the training samples.
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
            If X contains non-binary values, or if X and y have inconsistent numbers of samples.
            Also raised if weights are not provided when obj_mode is 'custom', or if the number
            of weights doesn't match the number of samples.

        Notes
        -----
        The behavior of this method depends on the `obj_mode` specified during initialization:
        - If obj_mode is 'acc', equal weights are used (weights parameter is ignored).
        - If obj_mode is 'balance', weights are automatically calculated to balance class importance.
        - If obj_mode is 'custom', the provided weights are used.

        When obj_mode is not 'custom' and weights are provided, a warning is issued and the weights are ignored.

        This method fits the BendersOCT model using Benders' decomposition approach.
        It sets up the master problem and subproblems, iteratively solves them,
        and generates Benders' cuts until convergence or the time limit is reached.

        The BendersOCT algorithm generally provides faster solution times compared to
        standard MIO formulations, especially for larger problem instances.

        Examples
        --------
        >>> from odtlearn.flow_oct import BendersOCT
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        >>> y = np.array([0, 1, 1, 0])
        >>> clf = BendersOCT(depth=2, time_limit=60)
        >>> clf.fit(X, y)
        """

        self._extract_metadata(X, y)

        check_binary(X)
        X, y = check_X_y(X, y)

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

        # we need these in the callback to have access to the value of the decision variables in the callback
        self._solver.store_data("g", self._g)
        self._solver.store_data("b", self._b)
        self._solver.store_data("p", self._p)
        self._solver.store_data("w", self._w)
        # We also pass the following information to the model as we need them in the callback
        # self._solver.model._self_obj = self
        self._solver.store_data("self", self)

        callback_action = BendersCallback

        self._solver.optimize(
            self._X,
            self,
            self._solver,
            callback=True,
            callback_action=callback_action,
            g=self._g,
            b=self._b,
            p=self._p,
            w=self._w,
        )

        self.b_value = self._solver.get_var_value(self._b, "b")
        self.w_value = self._solver.get_var_value(self._w, "w")
        self.p_value = self._solver.get_var_value(self._p, "p")

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X using the fitted BendersOCT model.

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
        >>> from odtlearn.flow_oct import BendersOCT
        >>> import numpy as np
        >>> X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        >>> y_train = np.array([0, 1, 1, 0])
        >>> clf = BendersOCT(depth=2)
        >>> clf.fit(X_train, y_train)
        >>> X_test = np.array([[1, 1], [0, 0]])
        >>> y_pred = clf.predict(X_test)
        >>> print(y_pred)
        [1 0]
        """
        # Check is fit had been called
        # for now we are assuming the model has been fit successfully if the fitted values for b, w, and p exist
        check_is_fitted(self, ["b_value", "w_value", "p_value"])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self._X_col_labels, X)

        return self._make_prediction(X)
