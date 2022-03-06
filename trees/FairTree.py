import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from trees.utils.StrongTreeUtils import check_binary, check_columns_match, get_predicted_value

# Include Tree.py, FlowOCT.py and BendersOCT.py in StrongTrees folder
from trees.utils.Tree import Tree
from trees.utils.StrongTreeFairOCT import FairOCT

from itertools import combinations

class FairTreeClassifier(ClassifierMixin, BaseEstimator):
    """Description of this estimator here


    Parameters
    ----------
    depth : int, default= 1
        A parameter specifying the depth of the tree
    time_limit : int, default= 30
        The given time limit (seconds) for solving the MIO in seconds
    _lambda : int, default= 0
        The regularization parameter in the objective
    num_threads: int, default=None
        The number of threads the solver should use. If None, it will use all avaiable threads
    positive_class : int
        The value of the class label which is corresponding to the desired outcome
    fairness_type: [None, 'SP', 'CSP', 'PE', 'EOpp', 'EOdds'], default=None
        The type of fairness we want to enforce
    fairness_bound: float (0,1], default=1
        The bound of the fairnes constraint. The smaller the value the stricter the fairness constraint and 1 corresponds to no fairness at all


    Examples
    --------
    >>> from trees.FairTree import FairTreeClassifier
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.random.randint(2, size=100)
    >>> P = np.arange(200).reshape(100, 2)
    >>> l = np.zeros((100, ))
    >>> fcl = FairTreeClassifier(positive_class = 1, depth = 1, _lambda = 0, time_limit = 10,
        fairness_type = 'CSP', fairness_bound = 1, num_threads = 1)
    >>> fcl.fit(X, y, P, l)
    """

    def __init__(
        self,
        positive_class,
        depth=1,
        _lambda=0,
        time_limit=30,
        fairness_type=None,
        fairness_bound=1,
        num_threads=None,
        obj_mode = 'acc'
    ):
        # this is where we will initialize the values we want users to provide
        self.depth = depth
        self.time_limit = time_limit
        self._lambda = _lambda
        self.num_threads = num_threads
        self.obj_mode = obj_mode 

        self.fairness_type = fairness_type
        self.fairness_bound = fairness_bound
        self.positive_class = positive_class

        self.X_col_labels = None
        self.X_col_dtypes = None
        self.y_dtypes = None

        self.P_col_labels = None
        self.P_col_dtypes = None
        self.l_dtypes = None

    def extract_metadata(self, X, y, P, l):
        """A function for extracting metadata from the inputs before converting
        them into numpy arrays to work with the sklearn API

        """
        if isinstance(X, pd.DataFrame):
            self.X_col_labels = X.columns
            self.X_col_dtypes = X.dtypes
        else:
            self.X_col_labels = np.array([f"X_{i}" for i in np.arange(0, X.shape[1])])

        if isinstance(P, pd.DataFrame):
            self.P_col_labels = P.columns
            self.P_col_dtypes = P.dtypes
        else:
            self.P_col_labels = np.array([f"P_{i}" for i in np.arange(0, P.shape[1])])

        self.labels = np.unique(y)


    def fit(self, X, y, P, l):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels in classification).
        P : array-like, shape (n_samples,1) or (n_samples, n_p)
            The protected feature columns (Race, gender, etc); We could have one or more columns

        l : array-like, shape (n_samples,)
            The legitimate factor column(e.g., prior number of criminal acts)

        Returns
        -------
        self : object
            Returns self.
        """
        # store column information and dtypes if any
        self.extract_metadata(X, y, P, l)
        # this function returns converted X and y but we retain metadata
        X, y = check_X_y(X, y)
        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)
        

        # Here we need to convert P and L to np.arrays. We need a function.
        # I am worried about the case if the shape is (n_samples, )
        P, l = check_X_y(P, l)

        # keep original data
        self.X_ = X
        self.y_ = y
        self.P_ = P
        self.l_ = l

        # Instantiate tree object here
        self.tree = Tree(self.depth)

        self.grb_model = FairOCT(
            X,
            y,
            self.tree,
            self.X_col_labels,
            self.labels,
            self._lambda,
            self.time_limit,
            self.num_threads,
            self.fairness_type,
            self.fairness_bound,
            self.positive_class,
            P,
            self.P_col_labels,
            l,
            self.obj_mode
        )
        self.grb_model.create_primal_problem()
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
        """A reference implementation of a prediction for a classifier.

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
        check_is_fitted(self, ["X_", "y_","P_", "l_"])

        if isinstance(X, pd.DataFrame):
            self.X_predict_col_names = X.columns
        else:
            self.X_predict_col_names = np.array([f"X_{i}" for i in np.arange(0, X.shape[1])])
        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self.X_col_labels, X)

        prediction = get_predicted_value(
            self.grb_model,
            X,
            self.b_value,
            self.w_value,
            self.p_value,
        )
        
        return prediction


    def get_SP(self, P, y):
        """
        This function returns the statistical parity value for any given protected level and outcome value

        :param P: array-like, shape (n_samples,1) or (n_samples, n_p)
                The protected feature columns (Race, gender, etc); We could have one or more columns
        :param y: array-like, shape (n_samples,)
                The target values (class labels in classification).
        

        :return sp_dict: a dictionary with key =(p,t) and value = P(Y=t|P=p) where p is a protected level and t is an outcome value

        """
        if isinstance(P, pd.DataFrame):
            self.P_test_col_names = P.columns
        else:
            self.P_test_col_names = np.array([f"P_{i}" for i in np.arange(0, P.shape[1])])

        
        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        P, y = check_X_y(P, y)

        check_columns_match(self.P_col_labels, P)

        class_name = "class_label"
        X_p = np.concatenate((P, y.reshape(-1, 1)), axis=1)
        X_p = pd.DataFrame(
            X_p,
            columns=(self.P_test_col_names.tolist() + [class_name]),
        )

        sp_dict = {}

        for t in X_p[class_name].unique():
            for protected_feature in self.P_test_col_names:
                for p in X_p[protected_feature].unique():
                    p_df = X_p[X_p[protected_feature] == p]
                    sp_p_t = None
                    if p_df.shape[0] != 0:
                        sp_p_t = p_df[p_df[class_name] == t].shape[0]/p_df.shape[0]
                    sp_dict[(p,t)] = sp_p_t

        return sp_dict

    def get_CSP(self, P, l, y):
        """
        This function returns the conditional statistical parity value for any given 
        protected level, legitimate feature value and outcome value

        :param P: array-like, shape (n_samples,1) or (n_samples, n_p)
                The protected feature columns (Race, gender, etc); We could have one or more columns
        :param l: array-like, shape (n_samples,)
            The legitimate factor column(e.g., prior number of criminal acts)
        :param y: array-like, shape (n_samples,)
                The target values (class labels in classification).
        

        :return csp_dict: a dictionary with key =(p, f, t) and value = P(Y=t|P=p, L=f) where p is a protected level
                          and t is an outcome value and l is the value of the legitimate feature

        """

        if isinstance(P, pd.DataFrame):
            self.P_test_col_names = P.columns
        else:
            self.P_test_col_names = np.array([f"P_{i}" for i in np.arange(0, P.shape[1])])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        _, y = check_X_y(P, y)
        P, l = check_X_y(P, l)

        check_columns_match(self.P_col_labels, P)

        class_name = "class_label"
        legitimate_name = "legitimate_feature_name"
        X_p = np.concatenate((P, l.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        X_p = pd.DataFrame(
            X_p,
            columns=(self.P_test_col_names.tolist() + [legitimate_name, class_name]),
        )

        csp_dict = {}

        for t in X_p[class_name].unique():
            for protected_feature in self.P_test_col_names:
                for p in X_p[protected_feature].unique():
                    for f in X_p[legitimate_name].unique():
                        p_f_df = X_p[(X_p[protected_feature] == p) & (X_p[legitimate_name] == f)]
                        csp_p_f_t = None
                        if p_f_df.shape[0] != 0:
                            csp_p_f_t = (p_f_df[p_f_df[class_name] == t].shape[0])/p_f_df.shape[0]
                        csp_dict[(p, f, t)] = csp_p_f_t

        return csp_dict

    def get_EqOdds(self, P, y, y_pred):
        """
        This function returns the false negative and true positive rate value 
        for any given protected level, outcome value and prediction value

        :param P: array-like, shape (n_samples,1) or (n_samples, n_p)
                The protected feature columns (Race, gender, etc); We could have one or more columns

        :param y: array-like, shape (n_samples,)
                The true target values (class labels in classification).
        :param y_pred: array-like, shape (n_samples,)
                The predicted values (class labels in classification).

        :return eq_dict: a dictionary with key =(p, t, t_pred) and value = P(Y_pred=t_pred|P=p, Y=t) 

        """

        if isinstance(P, pd.DataFrame):
            self.P_test_col_names = P.columns
        else:
            self.P_test_col_names = np.array([f"P_{i}" for i in np.arange(0, P.shape[1])])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        _, y = check_X_y(P, y)
        P, y_pred = check_X_y(P, y_pred)

        check_columns_match(self.P_col_labels, P)

        class_name = "class_label"
        pred_name = "pred_label"
        legitimate_name = "legitimate_feature_name"
        X_p = np.concatenate((P, y.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
        X_p = pd.DataFrame(
            X_p,
            columns=(self.P_test_col_names.tolist() + [class_name, pred_name]),
        )

        eq_dict = {}

        for t in X_p[class_name].unique():
            for t_pred in X_p[class_name].unique():
                for protected_feature in self.P_test_col_names:
                    for p in X_p[protected_feature].unique():
                        p_t_df = X_p[(X_p[protected_feature] == p) & (X_p[class_name] == t)]
                        eq_p_t_t_pred = None
                        if p_t_df.shape[0] != 0:
                            eq_p_t_t_pred = (p_t_df[p_t_df[pred_name] == t_pred].shape[0])/p_t_df.shape[0]
                        eq_dict[(p, t, t_pred)] = eq_p_t_t_pred

                    
                   


        return eq_dict

    def get_CondEqOdds(self, P, l, y, y_pred):
        """
        This function returns the conditional false negative and true positive rate value 
        for any given protected level, outcome value, prediction value and legitimate feature value

        :param P: array-like, shape (n_samples,1) or (n_samples, n_p)
                The protected feature columns (Race, gender, etc); We could have one or more columns
        :param l: array-like, shape (n_samples,)
            The legitimate factor column(e.g., prior number of criminal acts)

        :param y: array-like, shape (n_samples,)
                The true target values (class labels in classification).
        :param y_pred: array-like, shape (n_samples,)
                The predicted values (class labels in classification).

        :return ceq_dict: a dictionary with key =(p, f, t, t_pred) and value = P(Y_pred=t_pred|P=p, Y=t, L=f) 

        """

        if isinstance(P, pd.DataFrame):
            self.P_test_col_names = P.columns
        else:
            self.P_test_col_names = np.array([f"P_{i}" for i in np.arange(0, P.shape[1])])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        _, y = check_X_y(P, y)
        _, y_pred = check_X_y(P, y_pred)
        P, l = check_X_y(P, l)

        check_columns_match(self.P_col_labels, P)

        class_name = "class_label"
        pred_name = "pred_label"
        legitimate_name = "legitimate_feature_name"
        X_p = np.concatenate((P, l.reshape(-1, 1), y.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
        X_p = pd.DataFrame(
            X_p,
            columns=(self.P_test_col_names.tolist() + [legitimate_name, class_name, pred_name]),
        )

        ceq_dict = {}

        for t in X_p[class_name].unique():
            for t_pred in X_p[class_name].unique():
                for protected_feature in self.P_test_col_names:
                    for p in X_p[protected_feature].unique():
                        for f in X_p[legitimate_name].unique():
                            p_f_t_df = X_p[(X_p[protected_feature] == p) & (X_p[legitimate_name] == f) & (X_p[class_name] == t)]
                            ceq_p_f_t_t_pred = None
                            if p_f_t_df.shape[0] != 0:
                                ceq_p_f_t_t_pred = (p_f_t_df[p_f_t_df[pred_name] == t_pred].shape[0])/p_f_t_df.shape[0]
                            ceq_dict[(p, f, t, t_pred)] = ceq_p_f_t_t_pred

                    
                   


        return ceq_dict

