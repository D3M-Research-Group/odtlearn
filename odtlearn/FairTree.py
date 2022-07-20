import numpy as np
import pandas as pd

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from odtlearn.tree_classifier import TreeClassifier
from odtlearn.utils.strongtree_formulation import FairOCT
from odtlearn.utils.StrongTreeUtils import check_binary, check_columns_match
from odtlearn.utils.Tree import _Tree


class FairTreeClassifier(TreeClassifier):
    """An optimal and fair classification tree fitted on a given binary-valued
    data set. The fairness criteria enforced in the training step is one of statistical parity (SP),
    conditional statistical parity (CSP), predictive equality (PE), equal opportunity (EOpp) or equalized odds (EOdds).


    Parameters
    ----------
    depth : int, default= 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 60
        The given time limit (in seconds) for solving the MIO problem
    _lambda : float, default= 0
        The regularization parameter in the objective. _lambda is in the interval [0,1)
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    fairness_type: [None, 'SP', 'CSP', 'PE', 'EOpp', 'EOdds'], default=None
        The type of fairness criteria that we want to enforce
    fairness_bound: float (0,1], default=1
        The bound of the fairness constraint. The smaller the value the stricter
        the fairness constraint and 1 corresponds to no fairness at all

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    protect_feat_ : ndarray
        The protected feature columns passed during :meth: `fit`.
    legit_factor_ : ndarray
        The legitimate factor column passed during :meth: `fit`.
    b_value : a dictionary containing the value of the decision variables b,
    where b_value[(n,f)] is the value of b at node n and feature f
    w_value : a dictionary containing the value of the decision variables w,
    where w_value[(n,k)] is the value of w at node n and class label k
    p_value : a dictionary containing the value of the decision variables p,
    where p_value[n] is the value of p at node n
    grb_model : gurobipy.Model
        The fitted Gurobi model.

    Examples
    --------
    >>> from odtlearn.FairTree import FairTreeClassifier
    >>> import numpy as np
    >>> X = np.arange(200).reshape(100, 2)
    >>> y = np.random.randint(2, size=100)
    >>> protect_feat = np.arange(200).reshape(100, 2)
    >>> legit_factor = np.zeros((100, ))
    >>> fcl = FairTreeClassifier(positive_class = 1, depth = 1, _lambda = 0, time_limit = 60,
        fairness_type = 'CSP', fairness_bound = 1, num_threads = None)
    >>> fcl.fit(X, y, protect_feat, legit_factor)
    """

    def __init__(
        self,
        positive_class,
        depth=1,
        _lambda=0,
        time_limit=60,
        fairness_type=None,
        fairness_bound=1,
        num_threads=None,
        obj_mode="acc",
    ) -> None:
        super().__init__(depth, time_limit, num_threads)

        self._lambda = _lambda
        self.obj_mode = obj_mode

        self.fairness_type = fairness_type
        self.fairness_bound = fairness_bound
        self.positive_class = positive_class

        self.protect_feat_col_labels = None
        self.protect_feat_col_dtypes = None
        self.legit_factor_dtypes = None

    def fit(self, X, y, protect_feat, legit_factor, verbose=True):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels in classification).
        protect_feat : array-like, shape (n_samples,1) or (n_samples, n_p)
            The protected feature columns (Race, gender, etc); We could have one or more columns
        legit_factor : array-like, shape (n_samples,)
            The legitimate factor column(e.g., prior number of criminal acts)
        verbose : bool, default = True
            Flag for logging Gurobi outputs

        Returns
        -------
        self : object
            Returns self.
        """
        # store column information and dtypes if any
        self.extract_metadata(X, y, protect_feat=protect_feat)
        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)
        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)

        # Here we need to convert P and l to np.arrays.
        protect_feat, legit_factor = check_X_y(protect_feat, legit_factor)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # keep original data
        self.X_ = X
        self.y_ = y
        self.protect_feat_ = protect_feat
        self.legit_factor_ = legit_factor

        # Instantiate tree object here
        tree = _Tree(self.depth)

        self.grb_model = FairOCT(
            X,
            y,
            tree,
            self.X_col_labels,
            self.labels,
            self._lambda,
            self.fairness_type,
            self.fairness_bound,
            self.positive_class,
            protect_feat,
            self.protect_feat_col_labels,
            legit_factor,
            self.time_limit,
            self.num_threads,
            self.obj_mode,
            verbose,
        )
        self.grb_model.create_main_problem()
        self.grb_model.model.update()
        self.grb_model.model.optimize()

        # solving_time or other potential parameters of interest can be stored
        # within the class: self.solving_time
        self.solving_time = self.grb_model.model.getAttr("Runtime")

        # Here we will want to store these values and any other variables
        # needed for making predictions later
        self.b_value = self.grb_model.model.getAttr("X", self.grb_model.b)
        self.w_value = self.grb_model.model.getAttr("X", self.grb_model.w)
        self.p_value = self.grb_model.model.getAttr("X", self.grb_model.p)

        # Return the classifier
        return self

    def predict(self, X):
        """Classify test points using the FairTree classifier

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
        check_is_fitted(self, ["grb_model"])

        if isinstance(X, pd.DataFrame):
            self.X_predict_col_names = X.columns
        else:
            self.X_predict_col_names = np.array(
                [f"X_{i}" for i in np.arange(0, X.shape[1])]
            )
        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self.X_col_labels, X)

        prediction = self._get_prediction(X)

        return prediction

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
            self.protect_feat_test_col_names = protect_feat.columns
        else:
            self.protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        protect_feat, y = check_X_y(protect_feat, y)

        check_columns_match(self.protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        X_p = np.concatenate((protect_feat, y.reshape(-1, 1)), axis=1)
        X_p = pd.DataFrame(
            X_p,
            columns=(self.protect_feat_test_col_names.tolist() + [class_name]),
        )

        sp_dict = {}

        for t in X_p[class_name].unique():
            for protected_feature in self.protect_feat_test_col_names:
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
            self.protect_feat_test_col_names = protect_feat.columns
        else:
            self.protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        _, y = check_X_y(protect_feat, y)
        protect_feat, legit_factor = check_X_y(protect_feat, legit_factor)

        check_columns_match(self.protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        legitimate_name = "legitimate_feature_name"
        X_p = np.concatenate(
            (protect_feat, legit_factor.reshape(-1, 1), y.reshape(-1, 1)), axis=1
        )
        X_p = pd.DataFrame(
            X_p,
            columns=(
                self.protect_feat_test_col_names.tolist()
                + [legitimate_name, class_name]
            ),
        )

        csp_dict = {}

        for t in X_p[class_name].unique():
            for protected_feature in self.protect_feat_test_col_names:
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
            self.protect_feat_test_col_names = protect_feat.columns
        else:
            self.protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        _, y = check_X_y(protect_feat, y)
        protect_feat, y_pred = check_X_y(protect_feat, y_pred)

        check_columns_match(self.protect_feat_col_labels, protect_feat)

        class_name = "class_label"
        pred_name = "pred_label"
        X_p = np.concatenate(
            (protect_feat, y.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1
        )
        X_p = pd.DataFrame(
            X_p,
            columns=(
                self.protect_feat_test_col_names.tolist() + [class_name, pred_name]
            ),
        )

        eq_dict = {}

        for t in X_p[class_name].unique():
            for t_pred in X_p[class_name].unique():
                for protected_feature in self.protect_feat_test_col_names:
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
            self.protect_feat_test_col_names = protect_feat.columns
        else:
            self.protect_feat_test_col_names = np.array(
                [f"P_{i}" for i in np.arange(0, protect_feat.shape[1])]
            )

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        _, y = check_X_y(protect_feat, y)
        _, y_pred = check_X_y(protect_feat, y_pred)
        protect_feat, legit_factor = check_X_y(protect_feat, legit_factor)

        check_columns_match(self.protect_feat_col_labels, protect_feat)

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
                self.protect_feat_test_col_names.tolist()
                + [legitimate_name, class_name, pred_name]
            ),
        )

        ceq_dict = {}

        for t in X_p[class_name].unique():
            for t_pred in X_p[class_name].unique():
                for protected_feature in self.protect_feat_test_col_names:
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
        check_is_fitted(self, ["X_", "y_", "protected_feat_", "legit_factor_"])
        metric_names = ["SP", "CSP", "PE", "CPE"]
        if new_data is None:
            new_data = self.predict(self.X_)
        if metric not in metric_names:
            raise ValueError(
                f"metric argument: '{metric}' does not match any of the options: {metric_names}"
            )
        if metric == "SP":
            sp_df = pd.DataFrame(
                self.get_SP(self.protect_feat_, new_data).items(),
                columns=["(p,y)", "P(Y=y|P=p)"],
            )
            print(sp_df)
        elif metric == "CSP":
            csp_df = pd.DataFrame(
                self.get_CSP(self.protect_feat_, self.legit_factor_, new_data).items(),
                columns=["(p, f, y)", "P(Y=y|P=p, L=f)"],
            )
            print(csp_df)
        elif metric == "PE":
            pe_df = pd.DataFrame(
                self.get_EqOdds(self.protect_feat_, self.y_, new_data).items(),
                columns=["(p, y, y_pred)", "P(Y_pred=y_pred|P=p, Y=y)"],
            )
            print(pe_df)
        elif metric == "CPE":
            cpe_df = pd.DataFrame(
                self.get_CondEqOdds(
                    self.protect_feat_, self.legit_factor_, self.y_, new_data
                ).items(),
                columns=["(p, f, t, t_pred)" "P(Y_pred=y_pred|P=p, Y=y, L=f)"],
            )
            print(cpe_df)
