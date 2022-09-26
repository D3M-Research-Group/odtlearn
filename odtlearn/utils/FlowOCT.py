import numpy as np
import pandas as pd
from gurobipy import GRB, LinExpr
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from odtlearn.utils.FlowOCTSingleNode import FlowOCTSingleNode
from odtlearn.utils.validation import check_binary, check_columns_match


class FlowOCT(FlowOCTSingleNode):
    def __init__(
        self,
        _lambda=0,
        obj_mode="acc",
        depth=1,
        time_limit=60,
        num_threads=None,
        verbose=False,
    ) -> None:
        self.model_name = "FlowOCT"
        self.obj_mode = obj_mode
        super().__init__(
            _lambda,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    def _define_objective(self):
        ###########################################################
        # Define the Objective
        ###########################################################
        obj = LinExpr(0)
        for n in self.tree.Nodes:
            for f in self.X_col_labels:
                obj.add(-1 * self._lambda * self.b[n, f])
        if self.obj_mode == "acc":
            for i in self.datapoints:
                obj.add((1 - self._lambda) * self.z[i, 1])

        elif self.obj_mode == "balance":
            for i in self.datapoints:
                obj.add(
                    (1 - self._lambda)
                    * (1 / self.y[self.y == self.y[i]].shape[0] / self.labels.shape[0])
                    * self.z[i, 1]
                )
        else:
            assert self.obj_mode not in [
                "acc",
                "balance",
            ], "Wrong objective mode. obj_mode should be one of acc or balance."

        self.model.setObjective(obj, GRB.MAXIMIZE)

    def fit(self, X, y):

        # extract column labels, unique classes and store X as a DataFrame
        self._extract_metadata(X, y)

        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)
        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self._create_main_problem()
        self.model.update()
        self.model.optimize()

        self.b_value = self.model.getAttr("X", self.b)
        self.w_value = self.model.getAttr("X", self.w)
        self.p_value = self.model.getAttr("X", self.p)

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

        if isinstance(X, pd.DataFrame):
            self.X_predict_col_names = X.columns
        else:
            self.X_predict_col_names = np.arange(0, X.shape[1])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self.X_col_labels, X)

        return self._make_prediction(X)
