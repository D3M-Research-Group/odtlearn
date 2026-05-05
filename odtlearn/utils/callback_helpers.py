import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from pandas import DataFrame
from numpy import int64, str_
import heapq

from odtlearn.solvers.solver import Solver

# helper functions for BenderOCT callback

def benders_callback(model, X: DataFrame, obj, solver: Solver):
    g_trans, p_trans, b_trans, w_trans = (
            {k: solver.get_callback_solution(model, v) for k, v in obj._g.items()},
            {k: solver.get_callback_solution(model, v) for k, v in obj._p.items()},
            {k: solver.get_callback_solution(model, v) for k, v in obj._b.items()},
            {k: solver.get_callback_solution(model, v) for k, v in obj._w.items()},
        )

    for i in range(len(X)):
        g_threshold = 0.5
        if g_trans[i] > g_threshold:
            subproblem_value, left, right, target = benders_subproblem(
                obj, b_trans, p_trans, w_trans, i
            )
            if subproblem_value == 0:
                lhs = get_cut_integer(
                    solver,
                    obj,
                    left,
                    right,
                    target,
                    i,
                )
                solver.add_lazy_constraint(model, lhs)

def robust_benders_callback(model, X: DataFrame, obj, solver):
    t_trans, b_trans, w_trans = (
            {k: solver.get_callback_solution(model, v) for k, v in obj._t.items()},
            {k: solver.get_callback_solution(model, v) for k, v in obj._b.items()},
            {k: solver.get_callback_solution(model, v) for k, v in obj._w.items()},
        )
    # Initialize a blank-slate xi
    initial_xi = {}
    for c in obj._cat_features:
        initial_xi[c] = 0

    # Initialize dictionaries to store min and max values of each feature:
    initial_mins = {}
    initial_maxes = {}
    for c in obj._cat_features:
        initial_mins[c] = obj._min_values[c]
        initial_maxes[c] = obj._min_values[c]

    whole_expr = solver.lin_expr(0)  # Constraint RHS expression
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
    for i in obj._datapoints:
        nom_path, k = get_nominal_path(obj, b_trans, w_trans, i)
        if k != obj._y[i]:
            # Misclassified nominally - no need to check for shortest path
            curr_expr = get_cut_expression(
                obj,
                solver,
                X,
                b_trans,
                w_trans,
                nom_path,
                initial_xi,
                False,
                i,
                obj._f_theta_indices,
            )
            whole_expr += curr_expr
            new_constr = obj._t[i] <= curr_expr
            solver.add_lazy_constraint(model, new_constr)
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
    ) = get_all_terminal_paths(obj, b_trans, w_trans)
    for i in correct_points:
        path, xi, cost, v = robust_tree_subproblem(
            obj,
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
        if curr_cost + total_cost > obj._epsilon:
            # Push point back into queue
            heapq.heappush(priority_queue, current_point)
            break
        # Add RHS expression for point if still under budget
        i = current_point[1]
        whole_expr += get_cut_expression(
            obj,
            solver,
            X,
            b_trans,
            w_trans,
            path_dict[i],
            xi_dict[i],
            v_dict[i],
            i,
            obj._f_theta_indices,
        )
        total_cost += curr_cost

    added_cut = round(sum(t_trans)) > len(
        priority_queue
    )  # current sum of t is larger than RHS -> violated constraint(s)
    if added_cut:
        while len(priority_queue) != 0:
            current_point = heapq.heappop(priority_queue)
            i = current_point[1]
            whole_expr += get_cut_expression(
                obj,
                solver,
                obj._X,
                b_trans,
                w_trans,
                nom_path_dict[i],
                initial_xi,
                False,
                i,
                obj._f_theta_indices,
            )
        new_constr = (
            solver.quicksum(
                obj._t[i]
                for i in obj._datapoints
            )
            <= whole_expr
        )
        solver.add_lazy_constraint(model, new_constr)


def benders_subproblem(
    main_model_obj: "BendersOCT",  # noqa: F821
    b: Union[Dict[Tuple[int, str_], float], Dict[Tuple[int, str], float]],
    p: Dict[int, float],
    w: Dict[Tuple[int, int64], float],
    i: int,
) -> Union[
    Tuple[int, List[int], List[Any], List[int]],
    Tuple[int, List[Any], List[int], List[int]],
    Tuple[int, List[int], List[int], List[int]],
]:
    """
    Solve the Benders' subproblem for a given datapoint.

    Parameters
    ----------
    main_model_obj : object
        The main model object.
    b : dict
        The dictionary of binary decision variables representing the branching decisions.
    p : dict
        The dictionary of binary decision variables representing the prediction decisions.
    w : dict
        The dictionary of binary decision variables representing the class assignments.
    i : int
        The index of the current datapoint.

    Returns
    -------
    tuple
        A tuple containing the subproblem value, left nodes, right nodes, and target nodes.
    """
    label_i = main_model_obj._y[i]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        (
            _,
            branching,
            selected_feature,
            _,
            terminal,
            _,
        ) = main_model_obj._get_node_status(b, w, p, current)
        if terminal:
            target.append(current)
            if current in main_model_obj._tree.Nodes:
                left.append(current)
                right.append(current)
            if w[current, label_i] > 0.5:
                subproblem_value = 1
            break
        elif branching:
            if (
                main_model_obj._X.at[i, selected_feature] == 1
            ):  # going right on the branch
                left.append(current)
                target.append(current)
                current = main_model_obj._tree.get_right_children(current)
            else:  # going left on the branch
                right.append(current)
                target.append(current)
                current = main_model_obj._tree.get_left_children(current)

    return subproblem_value, left, right, target

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
    """
    Solve the robust tree subproblem for a given datapoint.

    The robust tree subproblem aims to find the shortest path from the root node to a terminal node
    that minimizes the cost while satisfying the robustness constraints. It takes into account the
    feature perturbations and label perturbations to ensure the robustness of the solution.

    Parameters
    ----------
    master : object
        The master problem object containing the problem data and variables.
    i : int
        The index of the current datapoint.
    terminal_nodes : list
        The list of terminal nodes in the decision tree.
    terminal_path_dict : dict
        A dictionary mapping each terminal node to its corresponding path from the root node.
    terminal_features_dict : dict
        A dictionary mapping each terminal node to the list of features encountered along its path.
    terminal_assignments_dict : dict
        A dictionary mapping each terminal node to its class assignment.
    terminal_cutoffs_dict : dict
        A dictionary mapping each terminal node to the list of cutoff values encountered along its path.
    initial_xi : dict, optional
        The initial dictionary of feature perturbations. Default is an empty dictionary.
    initial_mins : dict, optional
        The initial dictionary of minimum feature values. Default is an empty dictionary.
    initial_maxes : dict, optional
        The initial dictionary of maximum feature values. Default is an empty dictionary.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - target : list
            The list of nodes whose edges to the sink node are part of the minimum cut.
        - xi : dict
            The dictionary of feature perturbations that achieve the minimum cost.
        - cost : float
            The minimum cost of the robust tree subproblem.
        - v : bool
            A boolean indicating whether the label of the datapoint is perturbed in the optimal solution.

    """
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

def get_left_exp_integer(
    solver: Solver, main_grb_obj: "BendersOCT", n: int, i: int  # noqa: F821
):
    """
    Get the expression for the left branch constraint in the Benders' subproblem.

    Parameters
    ----------
    solver : Solver
        The solver object used for solving the optimization problem.
    main_grb_obj : object
        The main Gurobi model object.
    n : int
        The index of the current node.
    i : int
        The index of the current datapoint.

    Returns
    -------
    lhs : LinExpr
        The left-hand side expression of the left branch constraint.
    """
    lhs = solver.quicksum(
        -1 * main_grb_obj._b[n, f]
        for f in main_grb_obj._X_col_labels
        if main_grb_obj._X.at[i, f] == 0
    )

    return lhs


def get_right_exp_integer(
    solver: Solver, main_grb_obj: "BendersOCT", n: int, i: int  # noqa: F821
):
    """
    Get the expression for the right branch constraint in the Benders' subproblem.

    Parameters
    ----------
    solver : Solver
        The solver object used for solving the optimization problem.
    main_grb_obj : object
        The main Gurobi model object.
    n : int
        The index of the current node.
    i : int
        The index of the current datapoint.

    Returns
    -------
    lhs : LinExpr
        The left-hand side expression of the right branch constraint.
    """
    lhs = solver.quicksum(
        -1 * main_grb_obj._b[n, f]
        for f in main_grb_obj._X_col_labels
        if main_grb_obj._X.at[i, f] == 1
    )

    return lhs


def get_target_exp_integer(
    main_grb_obj: "BendersOCT", n: int, i: int  # noqa: F821
):
    """
    Get the expression for the target constraint in the Benders' subproblem.

    Parameters
    ----------
    main_grb_obj : object
        The main Gurobi model object.
    n : int
        The index of the current node.
    i : int
        The index of the current datapoint.

    Returns
    -------
    lhs : LinExpr
        The left-hand side expression of the target constraint.
    """
    label_i = main_grb_obj._y[i]
    lhs = -1 * main_grb_obj._w[n, label_i]
    return lhs


def get_cut_integer(
    solver: Solver,
    main_grb_obj: "BendersOCT",  # noqa: F821
    left: List[Union[Any, int]],
    right: List[Union[Any, int]],
    target: List[int],
    i: int,
):
    """
    Get the Benders' cut expression for the current subproblem.

    Parameters
    ----------
    solver : Solver
        The solver object used for solving the optimization problem.
    main_grb_obj : object
        The main Gurobi model object.
    left : list
        The list of nodes in the left branch of the current subproblem.
    right : list
        The list of nodes in the right branch of the current subproblem.
    target : list
        The list of target nodes in the current subproblem.
    i : int
        The index of the current datapoint.

    Returns
    -------
    lhs : LinExpr
        The left-hand side expression of the Benders' cut.
    """
    lhs = solver.lin_expr(0.0)
    lhs += main_grb_obj._g[i]
    for n in left:
        tmp_lhs = get_left_exp_integer(solver, main_grb_obj, n, i)
        lhs += tmp_lhs

    for n in right:
        tmp_lhs = get_right_exp_integer(solver, main_grb_obj, n, i)
        lhs += tmp_lhs

    for n in target:
        tmp_lhs = get_target_exp_integer(main_grb_obj, n, i)
        lhs += tmp_lhs

    return lhs <= 0


# helper functions for RobustTree callback


def get_cut_expression(master, solver, X, b, w, path, xi, v, i, f_theta_indices):
    """
    Get the cut expression for the RobustOCT subproblem.

    Parameters
    ----------
    master : object
        The master problem object.
    solver : Solver
        The solver object used for solving the optimization problem.
    X : DataFrame
        The input data.
    b : dict
        The dictionary of binary decision variables representing the branching decisions.
    w : dict
        The dictionary of binary decision variables representing the prediction decisions.
    path : list
        The current path in the tree.
    xi : dict
        The dictionary of feature perturbations.
    v : bool
        The label perturbation flag.
    i : int
        The index of the current datapoint.
    f_theta_indices : list
        The list of feature-threshold index pairs.

    Returns
    -------
    expr : LinExpr
        The cut expression for the current subproblem.
    """
    expr = solver.lin_expr(0)
    node_leaf_cutoff = np.power(
        2, master._tree.depth
    )  # anything at or above this number is a leaf

    # Add expressions to rhs where q = 1
    for x in range(len(path)):
        n = path[x]  # Current node
        if n < node_leaf_cutoff:
            if x == len(path) - 1:
                # Assigned a value at an internal node
                expr += solver.quicksum(
                    master._b[n, f, theta] for (f, theta) in f_theta_indices
                )
            # Add to expr if we went right according to our shortest path
            elif (2 * n) + 1 == path[x + 1]:
                expr += solver.quicksum(
                    master._b[n, f, theta]
                    for (f, theta) in f_theta_indices
                    if (X.at[i, f] + xi[f] <= theta)
                )
            # Add to expr if we went left according to our shortest path
            else:
                expr += solver.quicksum(
                    master._b[n, f, theta]
                    for (f, theta) in f_theta_indices
                    if (X.at[i, f] + xi[f] >= theta + 1)
                )
        # Add to expr the node going to the sink
        if not (x == len(path) - 1 and v):
            # Don't add edge to sink if at assignment node and label is changed
            expr += master._w[n, master._y[i]]
        else:
            expr += solver.quicksum(
                master._w[n, lab] for lab in master._labels if (lab != master._y[i])
            )
    return expr


def get_all_terminal_paths(
    master,
    b,
    w,
    terminal_nodes=[],
    path_dict={},
    feature_path_dict={},
    assignment_dict={},
    cutoff_dict={},
    curr_node=1,
    curr_path=[1],
    curr_feature_path=[],
    curr_cutoff_path=[],
):
    """
    Find all terminal paths in the decision tree.

    Parameters
    ----------
    master : object
        The master problem object.
    b : dict
        The dictionary of binary decision variables representing the branching decisions.
    w : dict
        The dictionary of binary decision variables representing the prediction decisions.
    terminal_nodes : list, optional
        The list of terminal nodes.
    path_dict : dict, optional
        The dictionary storing the paths to each terminal node.
    feature_path_dict : dict, optional
        The dictionary storing the feature paths to each terminal node.
    assignment_dict : dict, optional
        The dictionary storing the class assignments at each terminal node.
    cutoff_dict : dict, optional
        The dictionary storing the cutoff values along each path.
    curr_node : int, optional
        The current node being processed.
    curr_path : list, optional
        The current path being traversed.
    curr_feature_path : list, optional
        The current feature path being traversed.
    curr_cutoff_path : list, optional
        The current cutoff path being traversed.

    Returns
    -------
    tuple
        A tuple containing the updated terminal_nodes, path_dict, feature_path_dict,
        assignment_dict, and cutoff_dict.
    """
    new_path_dict = copy.deepcopy(path_dict)
    new_terminal_nodes = copy.deepcopy(terminal_nodes)
    new_feature_path_dict = copy.deepcopy(feature_path_dict)
    new_assignment_dict = copy.deepcopy(assignment_dict)
    new_cutoff_dict = copy.deepcopy(cutoff_dict)

    for k in master._labels:
        if w[curr_node, k] > 0.5:  # w[n,k] == 1
            # assignment node
            new_path_dict[curr_node] = curr_path
            new_terminal_nodes += [curr_node]
            new_feature_path_dict[curr_node] = curr_feature_path
            new_assignment_dict[curr_node] = k
            new_cutoff_dict[curr_node] = curr_cutoff_path
            return (
                new_terminal_nodes,
                new_path_dict,
                new_feature_path_dict,
                new_assignment_dict,
                new_cutoff_dict,
            )

    # b[n,f,theta]== 1
    curr_feature = None
    curr_theta = None
    for f, theta in master._f_theta_indices:
        if b[curr_node, f, theta] > 0.5:
            curr_feature = f
            curr_theta = theta
            break

    # Go left
    left_node = master._tree.get_left_children(curr_node)
    left_path = curr_path + [left_node]
    (
        left_terminal,
        left_paths,
        left_feature,
        left_assign,
        left_cutoff,
    ) = get_all_terminal_paths(
        master,
        b,
        w,
        terminal_nodes=terminal_nodes,
        path_dict=path_dict,
        feature_path_dict=feature_path_dict,
        assignment_dict=assignment_dict,
        cutoff_dict=cutoff_dict,
        curr_node=left_node,
        curr_path=left_path,
        curr_feature_path=curr_feature_path + [curr_feature],
        curr_cutoff_path=curr_cutoff_path + [curr_theta],
    )

    # Go right
    right_node = master._tree.get_right_children(curr_node)
    right_path = curr_path + [right_node]
    (
        right_terminal,
        right_paths,
        right_feature,
        right_assign,
        right_cutoff,
    ) = get_all_terminal_paths(
        master,
        b,
        w,
        terminal_nodes=terminal_nodes,
        path_dict=path_dict,
        feature_path_dict=feature_path_dict,
        assignment_dict=assignment_dict,
        cutoff_dict=cutoff_dict,
        curr_node=right_node,
        curr_path=right_path,
        curr_feature_path=curr_feature_path + [curr_feature],
        curr_cutoff_path=curr_cutoff_path + [curr_theta],
    )

    new_path_dict.update(left_paths)
    new_path_dict.update(right_paths)
    new_terminal_nodes += left_terminal
    new_terminal_nodes += right_terminal
    new_feature_path_dict.update(left_feature)
    new_feature_path_dict.update(right_feature)
    new_assignment_dict.update(left_assign)
    new_assignment_dict.update(right_assign)
    new_cutoff_dict.update(left_cutoff)
    new_cutoff_dict.update(right_cutoff)

    return (
        new_terminal_nodes,
        new_path_dict,
        new_feature_path_dict,
        new_assignment_dict,
        new_cutoff_dict,
    )


def get_nominal_path(master, b, w, i):
    """
    Get the nominal path for a correctly classified datapoint.

    Parameters
    ----------
    master : object
        The master problem object.
    b : dict
        The dictionary of binary decision variables representing the branching decisions.
    w : dict
        The dictionary of binary decision variables representing the prediction decisions.
    i : int
        The index of the current datapoint.

    Returns
    -------
    tuple
        A tuple containing the nominal path and the predicted class label.
    """
    path = []
    curr_node = 1

    while True:
        path += [curr_node]
        # Find whether a terminal node
        for k in master._labels:
            if w[curr_node, k] > 0.5:
                return path, k

        # braching node - find which feature to branch on
        for f, theta in master._f_theta_indices:
            if b[curr_node, f, theta] > 0.5:
                if master._X.at[i, f] >= theta + 1:
                    curr_node = (2 * curr_node) + 1  # go right
                else:
                    curr_node = 2 * curr_node  # go left
                break


def shortest_path_solver(
    master,
    i,
    label,
    terminal_nodes,
    terminal_path_dict,
    terminal_features_dict,
    terminal_assignments_dict,
    terminal_cutoffs_dict,
    initial_xi,
    initial_mins,
    initial_maxes,
):
    """
    Solve the shortest path problem for a given datapoint.

    Parameters
    ----------
    master : object
        The master problem object.
    i : int
        The index of the current datapoint.
    label : int
        The true class label of the datapoint.
    terminal_nodes : list
        The list of terminal nodes.
    terminal_path_dict : dict
        The dictionary storing the paths to each terminal node.
    terminal_features_dict : dict
        The dictionary storing the feature paths to each terminal node.
    terminal_assignments_dict : dict
        The dictionary storing the class assignments at each terminal node.
    terminal_cutoffs_dict : dict
        The dictionary storing the cutoff values along each path.
    initial_xi : dict
        The initial dictionary of feature perturbations.
    initial_mins : dict
        The initial dictionary of minimum feature values.
    initial_maxes : dict
        The initial dictionary of maximum feature values.

    Returns
    -------
    tuple
        A tuple containing the best path, best cost, feature perturbations (xi),
        and label perturbation flag (v).
    """
    best_cost = (master._epsilon + 1) * master._tree.depth
    best_path = []
    xi = copy.deepcopy(initial_xi)
    v = False

    for j in terminal_nodes:
        # Get cost of path
        curr_features = terminal_features_dict[j]
        curr_cutoffs = terminal_cutoffs_dict[j]
        curr_xi = copy.deepcopy(initial_xi)
        curr_v = terminal_assignments_dict[j] == label
        curr_mins = copy.deepcopy(initial_mins)
        curr_maxes = copy.deepcopy(initial_maxes)
        curr_path = terminal_path_dict[j]
        curr_cost = master._eta * int(
            curr_v
        )  # Start with cost if correctly classify point
        best_so_far = True
        for x in range(len(curr_path) - 1):
            n = curr_path[x]  # Current node
            f = curr_features[x]
            theta = curr_cutoffs[x]
            min_f = curr_mins[f]
            max_f = curr_maxes[f]

            curr_value = master._X.at[i, f] + curr_xi[f]
            # Went right
            if (2 * n) + 1 == curr_path[x + 1]:  # Path goes right
                if curr_value <= theta:
                    # See if can switch to go right by increasing x to theta+1
                    if max_f < theta + 1:
                        # Impossible path
                        best_so_far = False
                        break
                    # x + delta_x = theta + 1
                    delta_x = theta - master._X.at[i, f] + 1  # positive value

                    # cost increases by gamma per unit increase of xi
                    curr_cost += master._gammas.loc[i][f] * (
                        delta_x - curr_xi[f]
                    )
                    curr_xi[f] = delta_x

                # Update bounds
                curr_mins[f] = max(curr_mins[f], theta + 1)

            else:  # Went left
                if curr_value >= theta + 1:
                    # See if can switch to go left by decreasing x to theta
                    if min_f > theta:
                        # Impossible path
                        best_so_far = False
                        break
                    # x + delta_x = theta
                    delta_x = theta - master._X.at[i, f]  # negative value

                    # cost increases by gamma per unit decrease of xi
                    curr_cost += master._gammas.loc[i][f] * (
                        curr_xi[f] - delta_x
                    )
                    curr_xi[f] = delta_x

                # Update bounds
                curr_maxes[f] = min(curr_maxes[f], theta)

            if curr_cost > best_cost:
                # No need to go further
                best_so_far = False
                break
        if best_so_far:
            best_cost = curr_cost
            best_path = curr_path
            xi = curr_xi
            v = curr_v

    return best_path, best_cost, xi, v
