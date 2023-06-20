from abc import ABC, abstractmethod


class Solver(ABC):
    """
    Abstract base class for the solver interfaces.
    """

    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def get_var_value(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def add_vars(self):
        pass

    @abstractmethod
    def add_constrs(self):
        pass

    @abstractmethod
    def set_objective(self):
        pass

    @abstractmethod
    def lin_expr(self):
        pass

    @abstractmethod
    def quicksum(self):
        pass
