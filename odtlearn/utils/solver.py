from abc import ABC, abstractmethod

# Base classes for the different parts of an optimization problem
# Variables, Constraints, Objective, Model Container or Solver?


class Solver(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    def set_param(self, key, value):
        self.model.setParam(key, value)

    @abstractmethod
    def get_attr(self):
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
