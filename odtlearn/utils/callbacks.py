import copy
import heapq
import time

from gurobipy import GRB, LinExpr, quicksum

from odtlearn.utils.callback_helpers import (
    get_all_terminal_paths,
    get_cut_expression,
    get_cut_integer,
    get_nominal_path,
    shortest_path_solver,
)


def benders_subproblem(main_grb_obj, b, p, w, i):
    if "OPT" in type(main_grb_obj).__name__:
        label_i = main_grb_obj._t[i]
    else:
        label_i = main_grb_obj._y[i]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        _, branching, selected_feature, _, terminal, _ = main_grb_obj._get_node_status(
            b, w, p, current
        )
        if terminal:
            target.append(current)
            if current in main_grb_obj._tree.Nodes:
                left.append(current)
                right.append(current)
            if w[current, label_i] > 0.5:
                subproblem_value = 1
            break
        elif branching:
            if (
                main_grb_obj._X.at[i, selected_feature] == 1
            ):  # going right on the branch
                left.append(current)
                target.append(current)
                current = main_grb_obj._tree.get_right_children(current)
            else:  # going left on the branch
                right.append(current)
                target.append(current)
                current = main_grb_obj._tree.get_left_children(current)

    return subproblem_value, left, right, target


def benders_callback(model, where):
    """
    This function is called by Gurobi at every node through the branch-&-bound
    tree while we solve the model. Using the argument "where" we can see where
    the callback has been called.

    We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every data-point we solve
    the sub-problem which is a minimum cut and check if g[i] <= value of
    sub-problem[i]. If this is violated we add the corresponding benders
    constraint as lazy constraint to the master problem and proceed.
    Whenever we have no violated constraint, it means that we have found
    the optimal solution.

    :param model: the gurobi model we are solving.
    :param where: the node where the callback function is called from
    :return:
    """
    X = model._main_grb_obj._X

    if where == GRB.Callback.MIPSOL:
        func_start_time = time.time()
        model._callback_counter_integer += 1
        # we need the value of b, w and g
        g = model.cbGetSolution(model._vars_g)
        b = model.cbGetSolution(model._vars_b)
        p = model.cbGetSolution(model._vars_p)
        w = model.cbGetSolution(model._vars_w)

        added_cut = 0
        # We only want indices that g_i is one!
        for i in X.index:
            g_threshold = 0.5
            if g[i] > g_threshold:
                subproblem_value, left, right, target = benders_subproblem(
                    model._main_grb_obj, b, p, w, i
                )
                if subproblem_value == 0:
                    added_cut = 1
                    lhs = get_cut_integer(model._main_grb_obj, left, right, target, i)
                    model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time


def robust_tree_subproblem(
    master,
    i,
    terminal_nodes,
    terminal_path_dict,
    terminal_features_dict,
    terminal_assignments_dict,
    terminal_cutoffs_dict,
    initial_xi={},
    initial_mins={},
    initial_maxes={},
):
    label_i = master._y[i]
    target = []  # list of nodes whose edge to the sink is part of the min cut

    # Solve shortest path problem via DFS w/ pruning
    target, cost, xi, v = shortest_path_solver(
        master,
        i,
        label_i,
        terminal_nodes,
        terminal_path_dict,
        terminal_features_dict,
        terminal_assignments_dict,
        terminal_cutoffs_dict,
        initial_xi,
        initial_mins,
        initial_maxes,
    )

    return target, xi, cost, v


def robust_tree_callback(model, where):
    """
    This function is called by gurobi at every node through the branch-&-bound tree while we solve the model.
    Using the argument "where" we can see where the callback has been called.
    We are specifically interested at nodes where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every datapoint we solve the subproblem
    which is a minimum cut and check if g[i] <= value of subproblem[i].
    If this is violated we add the corresponding benders constraint as lazy constraint to the master
    problem and proceed. Whenever we have no violated constraint, it means that we have found the optimal solution.
    :param model: the gurobi model we are solving.
    :param where: the node where the callback function is called from
    :return:
    """
    if where == GRB.Callback.MIPSOL:
        func_start_time = time.time()
        model._callback_counter_integer += 1
        # we need the value of b, w, and t
        b = model.cbGetSolution(model._vars_b)
        w = model.cbGetSolution(model._vars_w)
        t = model.cbGetSolution(model._vars_t)

        # Initialize a blank-slate xi
        initial_xi = {}
        for c in model._master._cat_features:
            initial_xi[c] = 0

        # Initialize dictionaries to store min and max values of each feature:
        initial_mins = {}
        initial_maxes = {}
        for c in model._master._cat_features:
            initial_mins[c] = model._master._min_values[c]
            initial_maxes[c] = model._master._max_values[c]

        whole_expr = LinExpr(0)  # Constraint RHS expression
        priority_queue = []  # Stores elements of form (path_cost, index)
        path_dict = {}  # Stores all paths from shortest path problem
        xi_dict = (
            {}
        )  # Stores set of xi's from shortest path problem (feature perturbations)
        v_dict = (
            {}
        )  # Stores set of v's from shortest path problem (label perturbations)
        nom_path_dict = {}  # Stores set of nominal paths
        correct_points = []  # List of indices of nominally correctly classified points

        # Find nominal path for every data point
        for i in model._master._datapoints:
            nom_path, k = get_nominal_path(model._master, b, w, i)
            if k != model._master._y[i]:
                # Misclassified nominally - no need to check for shortest path
                curr_expr = get_cut_expression(
                    model._master, b, w, nom_path, initial_xi, False, i
                )
                whole_expr.add(curr_expr)
                model.cbLazy(model._master._t[i] <= curr_expr)
                model._total_cuts += 1
            else:
                # Correctly classified - put into pool of problems for shortest path
                correct_points += [i]
                nom_path_dict[i] = nom_path

        # Solve shortest path problem for every data point
        # Get all paths
        (
            terminal_nodes,
            terminal_path_dict,
            terminal_features_dict,
            terminal_assignments_dict,
            terminal_cutoffs_dict,
        ) = get_all_terminal_paths(model._master, b, w)
        for i in correct_points:
            path, xi, cost, v = robust_tree_subproblem(
                model._master,
                i,
                terminal_nodes,
                terminal_path_dict,
                terminal_features_dict,
                terminal_assignments_dict,
                terminal_cutoffs_dict,
                initial_xi=copy.deepcopy(initial_xi),
                initial_mins=copy.deepcopy(initial_mins),
                initial_maxes=copy.deepcopy(initial_maxes),
            )
            heapq.heappush(priority_queue, (cost, i))
            path_dict[i] = path
            xi_dict[i] = xi
            v_dict[i] = v

        # Add points that are misclassified to the constraint RHS
        total_cost = 0
        while True:
            if len(priority_queue) == 0:
                break

            # Get next least-cost point and see if still under epsilon budget
            current_point = heapq.heappop(priority_queue)
            curr_cost = current_point[0]
            if curr_cost + total_cost > model._master._epsilon:
                # Push point back into queue
                heapq.heappush(priority_queue, current_point)
                break
            # Add RHS expression for point if still under budget
            i = current_point[1]
            whole_expr.add(
                get_cut_expression(
                    model._master, b, w, path_dict[i], xi_dict[i], v_dict[i], i
                )
            )
            total_cost += curr_cost

        added_cut = round(sum(t)) > len(
            priority_queue
        )  # current sum of t is larger than RHS -> violated constraint(s)
        if added_cut:
            while len(priority_queue) != 0:
                current_point = heapq.heappop(priority_queue)
                i = current_point[1]
                whole_expr.add(
                    get_cut_expression(
                        model._master, b, w, nom_path_dict[i], initial_xi, False, i
                    )
                )
            model.cbLazy(
                quicksum(model._master._t[i] for i in model._master._datapoints)
                <= whole_expr
            )
            model._total_cuts += 1

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        model._total_callback_time_integer += func_time
        if added_cut:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time
