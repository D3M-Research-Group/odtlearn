from gurobipy import GRB, LinExpr
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from odtlearn.flow_oct_ss import FlowOCTSingleSink
from odtlearn.utils.callbacks import benders_callback
from odtlearn.utils.validation import check_binary, check_columns_match


class FlowOCT(FlowOCTSingleSink):
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
                obj.add(-1 * self._lambda * self._b[n, f])
        if self._obj_mode == "acc":
            for i in self._datapoints:
                obj.add((1 - self._lambda) * self._z[i, 1])

        elif self._obj_mode == "balance":
            for i in self._datapoints:
                obj.add(
                    (1 - self._lambda)
                    * (
                        1
                        / self._y[self._y == self._y[i]].shape[0]
                        / self._labels.shape[0]
                    )
                    * self._z[i, 1]
                )
        else:
            assert self._obj_mode not in [
                "acc",
                "balance",
            ], "Wrong objective mode. obj_mode should be one of acc or balance."
        self._solver.set_objective(obj, GRB.MAXIMIZE)

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
        self._solver.optimize()
        self.b_value = self._solver.get_attr("X", self._b)
        self.w_value = self._solver.get_attr("X", self._w)
        self.p_value = self._solver.get_attr("X", self._p)

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
    def __init__(
        self,
        _lambda=0,
        obj_mode="acc",
        depth=1,
        time_limit=60,
        num_threads=None,
        verbose=False,
    ) -> None:
        """
        Parameters
        ----------
        _lambda: The regularization parameter in the objective
        time_limit: The given time limit for solving the MIP
        num_threads: Specify number of threads for Gurobi to use when solving
        verbose: Display Gurobi model output
        """

        super().__init__(
            _lambda,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

        # The cuts we add in the callback function would be treated as lazy constraints
        self._model.params.LazyConstraints = 1
        """
        The following variables are used for the Benders problem to keep track
        of the times we call the callback.

        - counter_integer tracks number of times we call the callback from an
        integer node in the branch-&-bound tree
            - time_integer tracks the associated time spent in the
            callback for these calls
        - counter_general tracks number of times we call the callback from
        a non-integer node in the branch-&-bound tree
            - time_general tracks the associated time spent in the callback for
            these calls

        the ones ending with success are related to success calls.
        By success we mean ending up adding a lazy constraint
        to the model

        """
        self._model._total_callback_time_integer = 0
        self._model._total_callback_time_integer_success = 0

        self._model._total_callback_time_general = 0
        self._model._total_callback_time_general_success = 0

        self._model._callback_counter_integer = 0
        self._model._callback_counter_integer_success = 0

        self._model._callback_counter_general = 0
        self._model._callback_counter_general_success = 0

        # We also pass the following information to the model as we need them in the callback
        self._model._main_grb_obj = self

        self._lambda = _lambda
        self._obj_mode = obj_mode

    def _define_variables(self):
        self._tree_struc_variables()

        # g[i] is the objective value for the sub-problem[i]
        self._g = self._model.addVars(
            self._datapoints, vtype=GRB.CONTINUOUS, ub=1, name="g"
        )

        # we need these in the callback to have access to the value of the decision variables
        self._model._vars_g = self._g
        self._model._vars_b = self._b
        self._model._vars_p = self._p
        self._model._vars_w = self._w

    def _define_constraints(self):
        self._tree_structure_constraints()

    def _define_objective(self):
        obj = LinExpr(0)
        for n in self._tree.Nodes:
            for f in self._X_col_labels:
                obj.add(-1 * self._lambda * self._b[n, f])
        if self._obj_mode == "acc":
            for i in self._datapoints:
                obj.add((1 - self._lambda) * self._g[i])
        elif self._obj_mode == "balance":
            for i in self._datapoints:
                obj.add(
                    (1 - self._lambda)
                    * (
                        1
                        / self._y[self._y == self._y[i]].shape[0]
                        / self._labels.shape[0]
                    )
                    * self._g[i]
                )
        else:
            assert self._obj_mode not in [
                "acc",
                "balance",
            ], "Wrong objective mode. obj_mode should be one of acc or balance."

        self._model.setObjective(obj, GRB.MAXIMIZE)

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
        self._model.update()
        self._model.optimize(benders_callback)

        self.b_value = self._model.getAttr("X", self._b)
        self.w_value = self._model.getAttr("X", self._w)
        self.p_value = self._model.getAttr("X", self._p)

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
