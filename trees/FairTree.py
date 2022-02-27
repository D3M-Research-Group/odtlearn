import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from trees.utils.StrongTreeUtils import check_binary

# Include Tree.py, FlowOCT.py and BendersOCT.py in StrongTrees folder
from trees.utils.Tree import Tree
from trees.utils.StrongTreeFairOCT import FairOCT


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
    >>> y = np.zeros((100, ))
    >>> P = np.arange(200).reshape(100, 2)
    >>> l = np.zeros((100, ))
    >>> fcl = FairTreeClassifier(positive_class = 1, depth = 1, _lambda = 0, time_limit = 10,
        fairness_type = 'CSP', fairness_bound = 1, num_threads = 1)
    >>> fcl.fit(X_train, y_train, P, l)
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
    ):
        # this is where we will initialize the values we want users to provide
        self.depth = depth
        self.time_limit = time_limit
        self._lambda = _lambda
        self.num_threads = num_threads

        self.fairness_type = fairness_type
        self.fairness_bound = fairness_bound
        self.positive_class = positive_class

        self.X_col_labels = None
        self.X_col_dtypes = None
        self.y_dtypes = None

        self.P_col_labels = None
        self.P_col_dtypes = None
        self.l_col_dtypes = None

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

        self.y_dtypes = y.dtypes
        self.y_dtypes = l.dtypes
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
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

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

        self.primal = FairOCT(
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
        )
        self.primal.create_primal_problem()
        self.primal.model.update()
        self.primal.model.optimize()

        return self
