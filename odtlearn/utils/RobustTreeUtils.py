from gurobipy import GRB, LinExpr, quicksum
import time
import numpy as np
import pandas as pd
import copy
import heapq


def check_integer(df):
    if not np.array_equal(df.values, df.values.astype(int)):
        raise ValueError("Found non-integer values.")


def check_same_as_X(X, X_col_labels, G, G_label):
    """Check if a DataFrame G has the columns of X"""
    # Check if X has shape of G
    if X.shape[1] != G.shape[1]:
        raise ValueError(
            f"Input covariates has {X.shape[1]} columns but {G_label} has {G.shape[1]} columns"
        )

    # Check if X has same columns as G
    if isinstance(G, pd.DataFrame):
        if not np.array_equal(np.sort(X_col_labels), np.sort(G.columns)):
            raise KeyError(
                f"{G_label} should have the same columns as the input covariates"
            )
        return G
    else:
        # Check if X has default column labels or not
        if not np.array_equal(X_col_labels, np.arange(0, G.shape[1])):
            raise TypeError(
                f"{G_label} should be a Pandas DataFrame with the same columns as the input covariates"
            )
        return pd.DataFrame(G, columns=np.arange(0, G.shape[1]))


def get_cut_expression(master, b, w, path, xi, v, i):
    expr = LinExpr(0)
    node_leaf_cutoff = np.power(
        2, master.tree.depth
    )  # anything at or above this number is a leaf

    # Add expressions to rhs where q = 1
    for x in range(len(path)):
        n = path[x]  # Current node
        if n < node_leaf_cutoff:
            if x == len(path) - 1:
                # Assigned a value at an internal node
                expr += quicksum(
                    master.b[n, f, theta] for (f, theta) in master.f_theta_indices
                )
            # Add to expr if we went right according to our shortest path
            elif (2 * n) + 1 == path[x + 1]:
                expr += quicksum(
                    master.b[n, f, theta]
                    for (f, theta) in master.f_theta_indices
                    if (master.X.at[i, f] + xi[f] <= theta)
                )
            # Add to expr if we went left according to our shortest path
            else:
                expr += quicksum(
                    master.b[n, f, theta]
                    for (f, theta) in master.f_theta_indices
                    if (master.X.at[i, f] + xi[f] >= theta + 1)
                )
        # Add to expr the node going to the sink
        if not (x == len(path) - 1 and v):
            # Don't add edge to sink if at assignment node and label is changed
            expr += master.w[n, master.y[i]]
        else:
            expr += quicksum(
                master.w[n, lab] for lab in master.labels if (lab != master.y[i])
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
    """find all terminal paths"""
    new_path_dict = copy.deepcopy(path_dict)
    new_terminal_nodes = copy.deepcopy(terminal_nodes)
    new_feature_path_dict = copy.deepcopy(feature_path_dict)
    new_assignment_dict = copy.deepcopy(assignment_dict)
    new_cutoff_dict = copy.deepcopy(cutoff_dict)

    for k in master.labels:
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
    for (f, theta) in master.f_theta_indices:
        if b[curr_node, f, theta] > 0.5:
            curr_feature = f
            curr_theta = theta
            break

    # Go left
    left_node = master.tree.get_left_children(curr_node)
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
    right_node = master.tree.get_right_children(curr_node)
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
    """Get the nominal path for a correctly classified point"""
    path = []
    curr_node = 1

    while True:
        path += [curr_node]
        # Find whether a terminal node
        for k in master.labels:
            if w[curr_node, k] > 0.5:
                return path, k

        # braching node - find which feature to branch on
        for (f, theta) in master.f_theta_indices:
            if b[curr_node, f, theta] > 0.5:
                if master.X.at[i, f] >= theta + 1:
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
    best_cost = (master.epsilon + 1) * master.tree.depth
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
        curr_cost = master.eta * int(
            curr_v
        )  # Start with cost if correctly classify point
        best_so_far = True
        for x in range(len(curr_path) - 1):
            n = curr_path[x]  # Current node
            f = curr_features[x]
            theta = curr_cutoffs[x]
            min_f = curr_mins[f]
            max_f = curr_maxes[f]

            curr_value = master.X.at[i, f] + curr_xi[f]
            # Went right
            if (2 * n) + 1 == curr_path[x + 1]:  # Path goes right
                if curr_value <= theta:
                    # See if can switch to go right by increasing x to theta+1
                    if max_f < theta + 1:
                        # Impossible path
                        best_so_far = False
                        break
                    # x + delta_x = theta + 1
                    delta_x = theta - master.X.at[i, f] + 1  # positive value

                    # cost increases by gamma per unit increase of xi
                    curr_cost += master.gammas.loc[i][f] * (delta_x - curr_xi[f])
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
                    delta_x = theta - master.X.at[i, f]  # negative value

                    # cost increases by gamma per unit decrease of xi
                    curr_cost += master.gammas.loc[i][f] * (curr_xi[f] - delta_x)
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


def subproblem(
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
    label_i = master.y[i]
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


def mycallback(model, where):
    """
    This function is called by gurobi at every node through the branch-&-bound tree while we solve the model.
    Using the argument "where" we can see where the callback has been called. We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every datapoint we solve the subproblem which is a minimum cut and
    check if g[i] <= value of subproblem[i]. If this is violated we add the corresponding benders constraint as lazy
    constraint to the master problem and proceed. Whenever we have no violated constraint! It means that we have found
    the optimal solution.
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
        for c in model._master.cat_features:
            initial_xi[c] = 0

        # Initialize dictionaries to store min and max values of each feature:
        initial_mins = {}
        initial_maxes = {}
        for c in model._master.cat_features:
            initial_mins[c] = model._master.min_values[c]
            initial_maxes[c] = model._master.max_values[c]

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
        for i in model._master.datapoints:
            nom_path, k = get_nominal_path(model._master, b, w, i)
            if k != model._master.y[i]:
                # Misclassified nominally - no need to check for shortest path
                curr_expr = get_cut_expression(
                    model._master, b, w, nom_path, initial_xi, False, i
                )
                whole_expr.add(curr_expr)
                model.cbLazy(model._master.t[i] <= curr_expr)
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
            path, xi, cost, v = subproblem(
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
            if curr_cost + total_cost > model._master.epsilon:
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
                quicksum(model._master.t[i] for i in model._master.datapoints)
                <= whole_expr
            )
            model._total_cuts += 1

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        model._total_callback_time_integer += func_time
        if added_cut:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time
