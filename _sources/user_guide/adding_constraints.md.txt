# Adding side constraints to optimal classification trees
The purpose of this notebook is to demonstrate how users can leverage the object-oriented structure of the `ODTlearn` package to create optimal classification trees with side constraints. In particular, we will show how to construct an optimal classification tree with a one-sided fairness constraint instead of a two-sided fairness constraint used in `FairSPOCT`, `FairCSPOCT`, `FairPEOCT`, and `FairEOddsOCT` classes.

The figure below shows a simplified version of the class structure for optimal classification trees with fairness constraints. The `ConstrainedOCT` class includes an abstract method `_define_side_constraints()` which will be added to the optimization problem on top of the tree structure, flow, and arc constraints declared by the `FlowOCTMultipleSink` class. The `FairConstrainedOCT` class provides a method for adding a two-sided fairness constraint and the classes that inherit from it contain methods calculating the particular fairness notion of interest. However, for our example, we want a one-sided bound instead of a two-sided bound. In other words, we want the difference between two groups to not exceed a desired value rather than be bounded in absolute value.

<img src="../_static/img/constrained_class_diagram.jpg" alt="Simplified class diagram for ODTlearn" style="width:600px;display:block;margin-left:auto;margin-right:auto;"/>

To achieve this goal of creating an optimal classification tree with a one-sided fairness constraint, we will create a new class called `FairConstrainedOneSideOCT` that will inherit from `ConstrainedOCT`, and we will define a new method `_add_one_sided_fairness_constraint()` as shown in the class diagram below. While it is likely feasible to modify the `FairConstrainedOCT` class to allow users to specify whether they want a one-sided or two-sided fairness constraint when they initialize one of the four different `Fair*OCT` classes, such an approach leads to duplicated code and increased code complexity which makes future modifications to the code more complicated.

<img src="../_static/img/with_one_sided.jpg" alt="Simplified class diagram for ODTlearn with one-sided constrained class" style="width:600px;display:block;margin-left:auto;margin-right:auto;">


First we need to initialize the `FairConstrainedOneSideOCT` class and have it inherit from `ConstrainedOCT`.

```python
class FairConstrainedOneSideOCT(ConstrainedOCT):
    def __init__(
        self, solver, _lambda, depth, time_limit, num_threads, verbose
    ) -> None:
        super().__init__(solver, _lambda, depth, time_limit, num_threads, verbose)
```

Next, we will define our `_add_one_sided_fairness_constraint()` method which will take as arguments values of the fairness metric of interest for two groups $p$ and $p'$.

```python
    def _add_one_sided_fairness_constraint(self, p_df, p_prime_df):
        count_p = p_df.shape[0]
        count_p_prime = p_prime_df.shape[0]
        constraint_added = False
        if count_p != 0 and count_p_prime != 0:
            constraint_added = True
            self._solver.add_constr(
                (
                    (1 / count_p)
                    * self._solver.quicksum(
                        self._solver.quicksum(
                            self._zeta[i, n, self._positive_class]
                            for n in self._tree.Leaves + self._tree.Nodes
                        )
                        for i in p_df.index
                    )
                    - (
                        (1 / count_p_prime)
                        * self._solver.quicksum(
                            self._solver.quicksum(
                                self._zeta[i, n, self._positive_class]
                                for n in self._tree.Leaves + self._tree.Nodes
                            )
                            for i in p_prime_df.index
                        )
                    )
                )
                <= self._fairness_bound
            )

        return constraint_added
```

After our one-sided constraint method has been defined, we still need to define the remaining abstract methods: `_define_objective()`, `fit()`, and `predict()`. We omit these functions for this notebook, but for the interested reader, they would largely mirror the methods defined in `FairConstrained`.

With the `FairConstrainedOneSidedOCT` class completed, we can now define child classes for a fairness notion of interest. For this example, we will create a class for an optimal classification tree with a one-sided statistical parity constraint.

```python
class FairOneSideSPOCT(FairConstrainedOneSideOCT):
    def __init__(
        self,
        solver,
        positive_class,
        depth=1,
        time_limit=60,
        _lambda=0,
        obj_mode="acc",
        fairness_bound=1,
        num_threads=None,
        verbose=False,
    ) -> None:
        """
        An optimal classification tree fit on a given binary-valued data set
        with a one-sided fairness side-constraint requiring statistical parity (SP) between protected groups.

                Parameters
        ----------
        solver: str
            A string specifying the name of the solver to use
            to solve the MIP. Options are "Gurobi" and "CBC".
            If the CBC binaries are not found, Gurobi will be used by default.
        positive_class : int
            The value of the class label which is corresponding to the desired outcome
        depth : int, default = 1
            A parameter specifying the depth of the tree
        time_limit : int, default= 60
            The given time limit (in seconds) for solving the MIO problem
        _lambda : float, default = 0
            The regularization parameter in the objective. _lambda is in the interval [0,1)
        obj_mode: str, default="acc"
            The objective should be used to learn an optimal decision tree.
            The two options are "acc" and "balance".
            The accuracy objective attempts to maximize prediction accuracy while the
            balance objective aims to learn a balanced optimal decision
            tree to better generalize to our of sample data.
        fairness_bound: float (0,1], default=1
            The bound of the fairness constraint. The smaller the value the stricter
            the fairness constraint and 1 corresponds to no fairness constraint enforced
        num_threads: int, default=None
            The number of threads the solver should use. If None, it will use all avaiable threads

        """
        super().__init__(solver, _lambda, depth, time_limit, num_threads, verbose)

        self._obj_mode = obj_mode
        self._positive_class = positive_class
        self._fairness_bound = fairness_bound

    def _define_side_constraints(self):
        # Loop through all possible combinations of the protected feature
        for protected_feature in self._P_col_labels:
            for combo in combinations(self._X_p[protected_feature].unique(), 2):
                p = combo[0]
                p_prime = combo[1]

                p_df = self._X_p[self._X_p[protected_feature] == p]
                p_prime_df = self._X_p[self._X_p[protected_feature] == p_prime]
                self._add_fairness_constraint(p_df, p_prime_df)
```

The process for other notions of fairness including conditional statistical parity, equalized odds, and predictive equity is similar but with different formulas for calculating $p$ and $p'$. For additional details on how $p$ and $p'$ as well as further information about the fair optimal classification tree MIO formulation, please see the FairOCT paper: https://arxiv.org/abs/2201.09932
And with that we have added an extension to the `ODTlearn` package to 