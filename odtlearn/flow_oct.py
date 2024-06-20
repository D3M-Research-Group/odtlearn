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
    obj_mode : str, default="acc"
        The objective mode to be used for learning the optimal decision tree.
        Options are "acc" (accuracy) and "balance".
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
        self._obj_mode = obj_mode
        super().__init__(
            solver,
            _lambda,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_objective(self):
        obj = self._solver.lin_expr(0)
        # obj = LinExpr(0)
        for n in self._tree.Nodes:
            for f in self._X_col_labels:
                # obj.add(-1 * self._lambda * self._b[n, f])
                obj += -1 * self._lambda * self._b[n, f]
        assert self._obj_mode in [
            "acc",
            "balance",
        ], "Wrong objective mode. obj_mode should be one of acc or balance."
        if self._obj_mode == "acc":
            for i in self._datapoints:
                obj += (1 - self._lambda) * self._z[i, 1]

        else:
            for i in self._datapoints:
                obj += (
                    (1 - self._lambda)
                    * (
                        1
                        / self._y[self._y == self._y[i]].shape[0]
                        / self._labels.shape[0]
                    )
                    * self._z[i, 1]
                )
        self._solver.set_objective(obj, ODTL.MAXIMIZE)

    def fit(self, X, y):
        # extract column labels, unique classes and store X as a DataFrame
        self._extract_metadata(X, y)

        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)
        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self._classes = unique_labels(y)

        self._create_main_problem()
        self._solver.optimize(
            self._X, self, self._solver, callback=False, callback_action=None
        )
        self.b_value = self._solver.get_var_value(self._b, "b")
        self.w_value = self._solver.get_var_value(self._w, "w")
        self.p_value = self._solver.get_var_value(self._p, "p")

        return self

    def predict(self, X):
        """Classify test points using the StrongTree classifier

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
    obj_mode : str, default="acc"
        The objective mode to be used for learning the optimal decision tree.
        Options are "acc" (accuracy) and "balance".
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
        self._obj_mode = obj_mode

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
        assert self._obj_mode in [
            "acc",
            "balance",
        ], "Wrong objective mode. obj_mode should be one of acc or balance."
        if self._obj_mode == "acc":
            for i in self._datapoints:
                obj += (1 - self._lambda) * self._g[i]
        else:
            for i in self._datapoints:
                obj += (
                    (1 - self._lambda)
                    * (
                        1
                        / self._y[self._y == self._y[i]].shape[0]
                        / self._labels.shape[0]
                    )
                    * self._g[i]
                )

        self._solver.set_objective(obj, ODTL.MAXIMIZE)

    def fit(self, X, y):

        # extract column labels, unique classes and store X as a DataFrame
        self._extract_metadata(X, y)

        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)
        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self._classes = unique_labels(y)

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
        """Classify test points using the Benders' Formulation Classifier

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
        # Check is fit had been called
        # for now we are assuming the model has been fit successfully if the fitted values for b, w, and p exist
        check_is_fitted(self, ["b_value", "w_value", "p_value"])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self._X_col_labels, X)

        return self._make_prediction(X)
