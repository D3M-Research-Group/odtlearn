import copy

import numpy as np

# helper functions for BenderOCT callback


def get_left_exp_integer(solver, main_grb_obj, n, i):
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


def get_right_exp_integer(solver, main_grb_obj, n, i):
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


def get_target_exp_integer(main_grb_obj, n, i):
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


def get_cut_integer(solver, main_grb_obj, left, right, target, i):
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

    return lhs


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
    for f, theta in master._solver.model._data["f_theta_indices"]:
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
        for f, theta in master._solver.model._data["f_theta_indices"]:
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
    best_cost = (master._solver.model._data["epsilon"] + 1) * master._tree.depth
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
        curr_cost = master._solver.model._data["eta"] * int(
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
                    curr_cost += master._solver.model._data["gammas"].loc[i][f] * (
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
                    curr_cost += master._solver.model._data["gammas"].loc[i][f] * (
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
