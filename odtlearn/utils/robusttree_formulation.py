from gurobipy import LinExpr, GRB, quicksum
from odtlearn.utils.problem_formulation import ProblemFormulation


class RobustTreeFormulation(ProblemFormulation):
    def __init__(
        self,
        X,
        y,
        tree,
        X_col_labels,
        labels,
        costs,
        budget,
        model_name,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        super().__init__(
            X, y, tree, X_col_labels, model_name, time_limit, num_threads, verbose
        )
        self.cat_features = X_col_labels
        self.labels = labels
        # Regularization term: encourage less branching without sacrificing accuracy
        self.reg = 1 / (len(tree.Nodes) + 1)

        # Get range of data, and store indices of branching variables based on range
        min_values = X.min(axis=0)
        max_values = X.max(axis=0)
        f_theta_indices = []
        b_indices = []
        for f in self.cat_features:
            min_value = min_values[f]
            max_value = max_values[f]

            # cutoffs are from min_value to max_value - 1
            for theta in range(min_value, max_value):
                f_theta_indices += [(f, theta)]
                b_indices += [(n, f, theta) for n in self.tree.Nodes]

        self.min_values = X.min(axis=0)
        self.max_values = X.max(axis=0)
        self.f_theta_indices = f_theta_indices
        self.b_indices = b_indices

        # Create uncertainty set
        self.epsilon = budget  # Budget of uncertainty
        self.gammas = costs  # Cost of feature uncertainty
        self.eta = budget + 1  # Cost of label uncertainty - future work

        # Decision Variables
        self.t = 0
        self.b = 0
        self.w = 0
        # The cuts we add in the callback function would be treated as lazy constraints
        self.model.params.LazyConstraints = 1

        """
        The following variables are used for the Benders problem to keep track of the times we call the callback.

        - counter_integer tracks number of times we call the callback from an integer node
         in the branch-&-bound tree
            - time_integer tracks the associated time spent in the callback for these calls
        - counter_general tracks number of times we call the callback from a non-integer node
         in the branch-&-bound tree
            - time_general tracks the associated time spent in the callback for these calls

        the ones ending with success are related to success calls. By success we mean ending
        up adding a lazy constraint to the model
        """
        self.model._total_callback_time_integer = 0
        self.model._total_callback_time_integer_success = 0

        self.model._total_callback_time_general = 0
        self.model._total_callback_time_general_success = 0

        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

        self.model._callback_counter_general = 0
        self.model._callback_counter_general_success = 0

        self.model._total_cuts = 0

        # We also pass the following information to the model as we need them in the callback
        self.model._master = self


class RobustOCT(RobustTreeFormulation):
    def __init__(
        self,
        X,
        y,
        tree,
        X_col_labels,
        labels,
        costs,
        budget,
        model_name,
        time_limit,
        num_threads,
        verbose,
    ) -> None:
        self.model_name = "RobustOCT"
        super().__init__(
            X,
            y,
            tree,
            X_col_labels,
            labels,
            costs,
            budget,
            self.model_name,
            time_limit,
            num_threads,
            verbose,
        )

    ###########################################################
    # Create the master problem
    ###########################################################
    def create_main_problem(self):
        # define variables

        # t is the objective value of the problem
        self.t = self.model.addVars(
            self.datapoints, vtype=GRB.CONTINUOUS, ub=1, name="t"
        )
        # w[n,k] == 1 iff at node n we do not branch and we make the prediction k
        self.w = self.model.addVars(
            self.tree.Nodes + self.tree.Leaves, self.labels, vtype=GRB.BINARY, name="w"
        )

        # b[n,f,theta] ==1 iff at node n we branch on feature f with cutoff theta
        self.b = self.model.addVars(self.b_indices, vtype=GRB.BINARY, name="b")

        # we need these in the callback to have access to the value of the decision variables
        self.model._vars_t = self.t
        self.model._vars_b = self.b
        self.model._vars_w = self.w

        # define constraints

        # sum(b[n,f,theta], f, theta) + sum(w[n,k], k) = 1 for all n in nodes
        self.model.addConstrs(
            (
                quicksum(self.b[n, f, theta] for (f, theta) in self.f_theta_indices)
                + quicksum(self.w[n, k] for k in self.labels)
                == 1
            )
            for n in self.tree.Nodes
        )

        # sum(w[n,k], k) = 1 for all n in leaves
        self.model.addConstrs(
            (quicksum(self.w[n, k] for k in self.labels) == 1) for n in self.tree.Leaves
        )

        # define objective function
        obj = LinExpr(0)
        for i in self.datapoints:
            obj.add(self.t[i])
        # Add regularization term so that in case of tie in objective function,
        # encourage less branching
        obj.add(
            -1
            * self.reg
            * quicksum(self.b[n, f, theta] for (n, f, theta) in self.b_indices)
        )

        self.model.setObjective(obj, GRB.MAXIMIZE)
