'''
This module formulate the BendersOCT problem in gurobipy.
'''
from gurobipy import Model, GRB, quicksum, LinExpr
import numpy as np


class BendersOCT:
    def __init__(self, X, y, tree, X_col_labels, labels,
                 _lambda, time_limit, num_threads):
        '''
        :param X: numpy matrix or pandas dataframe of covariates
        :param y: numpy array or pandas series/dataframe of class labels
        :param tree: Tree object
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        '''
        self.X = X
        self.y = y

        self.X_col_labels = X_col_labels
        self.labels = labels
        # datapoints contains the indicies of our training data
        self.datapoints = np.arange(0, self.X.shape[0])

        self.tree = tree
        self._lambda = _lambda

        # Decision Variables
        self.g = 0
        self.b = 0
        self.p = 0
        self.w = 0


        # Gurobi model
        self.model = Model('BendersOCT')
        # The cuts we add in the callback function would be treated as lazy constraints
        self.model.params.LazyConstraints = 1
        self.model.params.Threads = num_threads
        self.model.params.TimeLimit = time_limit

        '''
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

        '''
        self.model._total_callback_time_integer = 0
        self.model._total_callback_time_integer_success = 0

        self.model._total_callback_time_general = 0
        self.model._total_callback_time_general_success = 0

        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

        self.model._callback_counter_general = 0
        self.model._callback_counter_general_success = 0

        # We also pass the following information to the model as we need them in the callback
        self.model._master = self


    def create_master_problem(self):
        '''
        This function create and return a gurobi model 
        formulating the BendersOCT problem
        :return:  gurobi model object with the BendersOCT formulation
        '''


        ###########################################################
        # Define Variables
        ###########################################################

        # g[i] is the objective value for the sub-problem[i]
        self.g = self.model.addVars(
            self.datapoints, vtype=GRB.CONTINUOUS, ub=1, name='g')
        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(
            self.tree.Nodes, self.X_col_labels, vtype=GRB.BINARY, name='b')
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name='p')
        # w[n,k]=1 iff at node n we predict class k
        self.w = self.model.addVars(self.tree.Nodes + self.tree.Leaves,
                                       self.labels, vtype=GRB.CONTINUOUS, lb=0,
                                       name='w')

        # we need these in the callback to have access to the value of the decision variables
        self.model._vars_g = self.g
        self.model._vars_b = self.b
        self.model._vars_p = self.p
        self.model._vars_w = self.w

        ###########################################################
        # Define Constraints
        ###########################################################

        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self.model.addConstrs(
            (quicksum(self.b[n, f] for f in self.X_col_labels) + self.p[n] + quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Nodes)


        # sum(w[n,k], k in labels) = p[n]
        self.model.addConstrs(
            (quicksum(self.w[n, k] for k in self.labels) == self.p[n]) for n in
            self.tree.Nodes + self.tree.Leaves)

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self.model.addConstrs(
            (self.p[n] + quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Leaves)

        ###########################################################
        # Define the Objective
        ###########################################################
        obj = LinExpr(0)
        for i in self.datapoints:
            obj.add((1 - self._lambda) * (self.g[i] - 1))

        for n in self.tree.Nodes:
            for f in self.X_col_labels:
                obj.add(-1 * self._lambda * self.b[n, f])

        self.model.setObjective(obj, GRB.MAXIMIZE)
