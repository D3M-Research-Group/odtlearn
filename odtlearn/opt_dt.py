from abc import ABC, abstractmethod

from gurobipy import Model

from odtlearn.utils.Tree import _Tree


class OptimalDecisionTree(ABC):
    def __init__(self, depth=1, time_limit=60, num_threads=None, verbose=False) -> None:
        """
        Parameters
        ----------
        X : ndarray, pd.DataFrame
            Matrix of covariates
        y : ndarray, pd.Seres
            Array of class labels
        depth : int, default=1
            A parameter specifying the depth of the tree
        time_limit : int, default=60
            The given time limit (in seconds) for solving the MIO problem
        num_threads: int, default=None
            The number of threads the solver should use. If no argument is supplied,
            Gurobi will use all available threads.

        """
        # self.X = pd.DataFrame(X, columns=X_col_labels)
        # self.y = y
        # self.X_col_labels = X_col_labels

        self._depth = depth
        self._time_limit = time_limit
        self._num_threads = num_threads
        self._verbose = verbose

        # decision variables
        self._b = 0
        self._p = 0
        self._w = 0
        self._zeta = 0  # rename?
        self._z = 0

        # datapoints contains the indicies of our training data
        # self.datapoints = np.arange(0, self.X.shape[0])

        self._tree = _Tree(self._depth)
        self._time_limit = time_limit
        # self.model_name = model_name
        # Gurobi model
        self._model = Model()
        if not verbose:
            # supress all logging
            self._model.params.OutputFlag = 0
        if num_threads is not None:
            self._model.params.Threads = num_threads
        self._model.params.TimeLimit = time_limit

    @abstractmethod
    def _define_variables(self):
        pass

    @abstractmethod
    def _define_constraints(self):
        pass

    @abstractmethod
    def _define_objective(self):
        pass

    def _create_main_problem(self):
        """
        This function creates and return a gurobi model based on the
        variables, constraints, and objective defined within a subclass
        """
        self._define_variables()
        self._define_constraints()
        self._define_objective()

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass
