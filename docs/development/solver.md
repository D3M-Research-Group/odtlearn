# Adding Additional Solvers to ODTLearn

This document covers how to add new MIO solvers to ODTLearn. Currently, ODTLearn supports CBC through Python MIP and Gurobi.

Please read the "How to contribute to ODTlearn" document before moving forward.


## Implementation
The `Solver` abstract base class provides a common inteface for all solvers in ODTLearn. For examples of how to extend the `Solver` class to integrate a new solver, the `CBCSolver` and `GurobiSolver` classes can be referred to as examples. Detailed steps to integrate new solvers are below.

### Dependencies
Any dependencies related to a new solver should be optional. Ensure that `pyproject.toml` lists any added dependencies needed for a new solver as an optional dependency.

### The `Solver` Abstract Base Class

In order to integrate a new MIO solver to ODTLearn, a wrapper class should be created that inherits from `Solver`, implementing all of its abstract methods. The naming convention used in ODTLearn for this new class is `<XX>Solver`, for `<XX>` the solver name (e.g., Gurobi, CBC). This solver should exist in its own file in the `odtlearn/solvers` folder with the filename `<xx>_solver.py`.

The specifics of each abstract method can be found in the API documentation of the `Solver` class, and are summarized in the following.

* Methods to build the model and run the solver
    * `add_vars()` - A method to add variables to the MIO model
    * `add_constr()` - A method to add constraints to the MIO model
    * `set_objective()` - A method to set the objective function of the MIO model
    * `set_time_limit()` - A method to set the solver time limit of the model
    * `set_num_threads()`- A method to set the number of threads used in the solver
    * `get_num_variables()` - Gets the number of variables in the model.
    * `get_num_constraints()` - Gets the number of constraints in the model.
    * `optimize()` - A method that starts the solver optimization.
* Methods to retrieve solve information after solving
    * `get_var_value()` - Retrieves the optimal solution value after solving.
    * `get_gap()` - Retrieves the optimality gap of the MIO solver after solving.
    * `get_objective_value()` - Retrieves the optimal objective value after solving.
    * `get_objective_bound()` - Retrieves the MIO objective bounds after solving.
    * `get_num_int_variables()` - Retrieves the number of integer variables in the solution after solving.
    * `get_num_nonzeroes()` - Retrieves the number of nonzero variables in the solution after solving.
    * `get_num_soluions()` - Retrieves the number of solutions found during the MIO solve.
    * `get_search_progress_log()` - Gets the search progress log. The search progress log is a list of tuples of the form (time, (lower bound, upper bound)). Additional helper methods can be implemented to ensure that the progress log is saved into the Solver object in this form during the solve if `search_progress_log` is set to true.
* Methods to build solver-specific representations of linear expressions
    * `lin_expr()` - Creates the solver-specific object that represents linear expressions. For instance, it returns `mip.LinExpr` with CBC and `gurobipy.LinExpr` with Gurobi.
    * `quicksum()` - Adds variables together to return the solver-specific object that represents linear expressions. For instance, it wraps around `mip.xsum` with CBC and `gurobipy.quicksum` with Gurobi.
* Methods to set and use callbacks (for `BendersOCT` and `RobustOCT`)
    * `set_callback()` - A method to set callbacks in the solver (if supported)
    * `store_data()` - A method that can store data needed to access to callbacks. This can be done by adding an attribute `_data` of type `dict` to the solver's model object, allowing for any key-value pair to be stored.
    * `add_lazy_constraint()` - 
    * `get_callback_solution()`

If a solver does not support callbacks, then these four functions should throw an error. Note that this would mean that `BendersOCT` and `RobustOCT` would not be supported by this added solver, which should be specified in the documentation and with an error thrown if trying to fit a `BendersOCT` or `RobustOCT` object with the solver.

We detail callback support in the following section.

### Callbacks
Callbacks are necessary to use the `BendersOCT` and `RobustOCT` formulations in ODTLearn. Not all MIO solvers support callbacks, and thus `BendersOCT` and `RobustOCT` may not be usable in a new MIO solver. An error should be thrown if a user attempts to instantiate a `BendersOCT` or `RobustOCT` object with a solver without callback support.

Callbacks should be specified in a file within `odtlearn/solvers`, with a filename `<xx>_callbacks.py` for `<xx>` the solver name.

* For `BendersOCT`, the helper function `benders_callback` in `odtlearn/utils/callback_helpers.py` should be used to generate the appropriate Benders cut to be added to the model.
* For `RobustOCT`, the helper function `robust_benders_callback` in `odtlearn/utils/callback_helpers.py` should be used to generate the appropriate Benders cut to be added to the model.

### Adding the new solver as an option

In the constructor of the abstract base class `OptimalDecisionTree` in `odtlearn/opt.dt.py`, set `self._solver` to be the new `Solver` object corresponding to the new solver. This should be in the following general format (with "new" replaced by the new solver name):
```
if solver.lower() == "new":
    try:
        from odtlearn.solvers.new_solver import NewSolver

        self._solver = NewSolver(verbose)
    except ModuleNotFoundError as e:
        raise ImportError(
            "New solver requires the 'new' package, which is not installed. "
            "Install it with: [pip install new]. "
            "You must also have a valid New license."
        ) from e
```

## Testing
Tests can be added to ODTLearn in `odtlearn/tests`. To test the functions in the extended `Solver` class, tests should be added to `odtlearn/tests/test_solvers.py`

To ensure expected behavior during optimization, `test_FairOCT.py`, `test_FairOCT.py`, `test_Flow_Benders_OCT.py`, `test_FlowOPT.py`, and `test_RobustOCT.py`should be modified in `odtlearn/tests` to add tests for the specific solver. 

To properly build such tests in Github Actions, ensure that `test` in `pyproject.toml` is updated with the necessary dependencies.

## Updating Documentation
 If not inherited from the `Solver` abstract base class, the new extended `Solver` class and methods should have docstrings.

 In addition, update `docs/installation.md` by specifying how to install the new solver.

