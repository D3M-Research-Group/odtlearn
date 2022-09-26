from abc import ABC, abstractmethod

from gurobipy import Model

from odtlearn.utils.Tree import _Tree


class ProblemFormulation(ABC):
    def __init__(self, depth=1, time_limit=60, num_threads=None, verbose=False) -> None:
        """
        :param X: numpy matrix of covariates
        :param y: numpy array of class labels
        :param X_col_labels: a list of features in the covariate space X
        :param model_name: str name of Gurobi model

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

        self.depth = depth
        self.time_limit = time_limit
        self.num_threads = num_threads
        self.verbose = verbose

        # decision variables
        self.b = 0
        self.p = 0
        self.w = 0
        self.zeta = 0  # rename?
        self.z = 0

        # datapoints contains the indicies of our training data
        # self.datapoints = np.arange(0, self.X.shape[0])

        self.tree = _Tree(self.depth)
        self.time_limit = time_limit
        # self.model_name = model_name
        # Gurobi model
        self.model = Model()
        if not verbose:
            # supress all logging
            self.model.params.OutputFlag = 0
        if num_threads is not None:
            self.model.params.Threads = num_threads
        self.model.params.TimeLimit = time_limit

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

    def print_tree(self):
        pass

    def plot_tree(self):
        pass
