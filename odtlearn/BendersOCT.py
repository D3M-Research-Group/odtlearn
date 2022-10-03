from gurobipy import GRB, LinExpr, quicksum
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from odtlearn.opt_ct import OptimalClassificationTree
from odtlearn.utils.callbacks import benders_callback
from odtlearn.utils.validation import check_binary, check_columns_match


class BendersOCT(OptimalClassificationTree):
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
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        :param num_threads: Specify number of threads for Gurobi to use when solving
        :param verbose: Display Gurobi model output
        """

        super().__init__(
            depth,
            time_limit,
            num_threads,
            verbose,
        )

        self._g = 0

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
        self.obj_mode = obj_mode

    def _define_variables(self):
        ###########################################################
        # Define Variables
        ###########################################################

        # g[i] is the objective value for the sub-problem[i]
        self._g = self._model.addVars(
            self._datapoints, vtype=GRB.CONTINUOUS, ub=1, name="g"
        )
        # b[n,f] ==1 iff at node n we branch on feature f
        self._b = self._model.addVars(
            self._tree.Nodes, self._X_col_labels, vtype=GRB.BINARY, name="b"
        )
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self._p = self._model.addVars(
            self._tree.Nodes + self._tree.Leaves, vtype=GRB.BINARY, name="p"
        )
        # w[n,k]=1 iff at node n we predict class k
        self._w = self._model.addVars(
            self._tree.Nodes + self._tree.Leaves,
            self._labels,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="w",
        )

        # we need these in the callback to have access to the value of the decision variables
        self._model._vars_g = self._g
        self._model._vars_b = self._b
        self._model._vars_p = self._p
        self._model._vars_w = self._w

    def _define_constraints(self):
        ###########################################################
        # Define Constraints
        ###########################################################

        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self._model.addConstrs(
            (
                quicksum(self._b[n, f] for f in self._X_col_labels)
                + self._p[n]
                + quicksum(self._p[m] for m in self._tree.get_ancestors(n))
                == 1
            )
            for n in self._tree.Nodes
        )

        # sum(w[n,k], k in labels) = p[n]
        self._model.addConstrs(
            (quicksum(self._w[n, k] for k in self._labels) == self._p[n])
            for n in self._tree.Nodes + self._tree.Leaves
        )

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self._model.addConstrs(
            (
                self._p[n] + quicksum(self._p[m] for m in self._tree.get_ancestors(n))
                == 1
            )
            for n in self._tree.Leaves
        )

    def _define_objective(self):
        ###########################################################
        # Define the Objective
        ###########################################################
        obj = LinExpr(0)
        for n in self._tree.Nodes:
            for f in self._X_col_labels:
                obj.add(-1 * self._lambda * self._b[n, f])
        if self.obj_mode == "acc":
            for i in self._datapoints:
                obj.add((1 - self._lambda) * self._g[i])
        elif self.obj_mode == "balance":
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
            assert self.obj_mode not in [
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
