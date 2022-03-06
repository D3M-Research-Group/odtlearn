import numpy as np
import pandas as pd
import time
from gurobipy import LinExpr, quicksum, GRB


def check_columns_match(original_columns, new_data):
    """
    :param original_columns: List of column names from the data set used to fit the model
    :param new_data: The numpy matrix or pd dataframe new data set for
    which we want to make predictions
    :return ValueError if column names do not match, otherwise None
    """

    if isinstance(new_data, pd.DataFrame):
        new_column_names = new_data.columns
        # take difference of sets
        non_matched_columns = set(new_column_names) - set(original_columns)
        if len(non_matched_columns) > 0:
            raise ValueError(
                f"Columns {list(non_matched_columns)} found in prediction data, but not found in fit data."
            )
    else:
        # we are assuming the order of columns matches and we will just check that shapes match
        # assuming here that new_data is a numpy matrix
        assert (
            len(original_columns) != new_data.shape[0]
        ), f"Fit data has {len(original_columns)} columns but new data has {new_data.shape[0]} columns."


def check_binary(df):
    # TO-DO: truncate output if lots of non_binary_columns
    if isinstance(df, pd.DataFrame):
        non_binary_columns = [
            col for col in df if not np.isin(df[col].dropna().unique(), [0, 1]).all()
        ]
        if len(non_binary_columns) > 0:
            raise ValueError(
                f"Found columns ({non_binary_columns}) that contain values other than 0 or 1."
            )
    else:
        assert (
            (df == 0) | (df == 1)
        ).all(), "Expecting all values of covariate matrix to be either 0 or 1."


def get_node_status(grb_model, b, w, p, n):
    """
    This function give the status of a given node in a tree. By status we mean whether the node
        1- is pruned? i.e., we have made a prediction at one of its ancestors
        2- is a branching node? If yes, what feature do we branch on
        3- is a leaf? If yes, what is the prediction at this node?
    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param w: The values of prediction decision variable w
    :param p: The values of decision variable p
    :param n: A valid node index in the tree
    :return: pruned, branching, selected_feature, leaf, value
    pruned=1 iff the node is pruned
    branching = 1 iff the node branches at some feature f
    selected_feature: The feature that the node branch on
    leaf = 1 iff node n is a leaf in the tree
    value: if node n is a leaf, value represent the prediction at this node
    """

    pruned = False
    branching = False
    leaf = False
    value = None
    selected_feature = None

    p_sum = 0
    for m in grb_model.tree.get_ancestors(n):
        p_sum = p_sum + p[m]
    if p[n] > 0.5:  # leaf
        leaf = True
        for k in grb_model.labels:
            if w[n, k] > 0.5:
                value = k
    elif p_sum == 1:  # Pruned
        pruned = True

    if n in grb_model.tree.Nodes:
        if (pruned is False) and (leaf is False):  # branching
            for f in grb_model.X_col_labels:
                if b[n, f] > 0.5:
                    selected_feature = f
                    branching = True

    return pruned, branching, selected_feature, leaf, value


def print_tree(grb_model, b, w, p):
    """
    This function print the derived tree with the branching features and the predictions asserted for each node
    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param w: The values of prediction decision variable w
    :param p: The values of decision variable p
    :return: print out the tree in the console
    """
    for n in grb_model.tree.Nodes + grb_model.tree.Leaves:
        pruned, branching, selected_feature, leaf, value = get_node_status(
            grb_model, b, w, p, n
        )
        print("#########node ", n)
        if pruned:
            print("pruned")
        elif branching:
            print("branch on {}".format(selected_feature))
        elif leaf:
            print("leaf {}".format(value))


def get_predicted_value(grb_model, X, b, w, p):
    """
    This function returns the predicted value for a given dataset
    :param grb_model: The gurobi model we solved
    :param X: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param w: The value of decision variable w
    :param p: The value of decision variable p
    :return: The predicted value for dataset X
    """
    predicted_values = []
    for i in range(X.shape[0]):
        current = 1
        while True:
            _, branching, selected_feature, leaf, value = get_node_status(
                grb_model, b, w, p, current
            )
            if leaf:
                predicted_values.append(value)
                break
            elif branching:
                selected_feature_idx = np.where(grb_model.X_col_labels == selected_feature)
                # Raise assertion error we don't have a column that matches
                # the selected feature or more than one column that matches
                assert (
                    len(selected_feature_idx) == 1
                ), f"Found {len(selected_feature_idx)} columns matching the selected feature {selected_feature}"
                if X[i, selected_feature_idx] == 1:  # going right on the branch
                    current = grb_model.tree.get_right_children(current)
                else:  # going left on the branch
                    current = grb_model.tree.get_left_children(current)
    return np.array(predicted_values)


def get_left_exp_integer(main_grb_obj, n, i):
    lhs = quicksum(
        -1*main_grb_obj.b[n, f]
        for f in main_grb_obj.X_col_labels
        if main_grb_obj.X.at[i, f] == 0
    )

    return lhs


def get_right_exp_integer(main_grb_obj, n, i):
    lhs = quicksum(
        -1*main_grb_obj.b[n, f]
        for f in main_grb_obj.X_col_labels
        if main_grb_obj.X.at[i, f] == 1
    )

    return lhs


def get_target_exp_integer(main_grb_obj, n, i):
    label_i = main_grb_obj.y[i]
    lhs = -1 * main_grb_obj.w[n, label_i]
    return lhs


def get_cut_integer(main_grb_obj, left, right, target, i):
    lhs = LinExpr(0) + main_grb_obj.g[i]
    for n in left:
        tmp_lhs = get_left_exp_integer(main_grb_obj, n, i)
        lhs = lhs + tmp_lhs

    for n in right:
        tmp_lhs = get_right_exp_integer(main_grb_obj, n, i)
        lhs = lhs + tmp_lhs

    for n in target:
        tmp_lhs = get_target_exp_integer(main_grb_obj, n, i)
        lhs = lhs + tmp_lhs

    return lhs


def subproblem(main_grb_obj, b, p, w, i):
    label_i = main_grb_obj.y[i]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        _, branching, selected_feature, terminal, _ = get_node_status(
            main_grb_obj, b, w, p, current
        )
        if terminal:
            target.append(current)
            if current in main_grb_obj.tree.Nodes:
                left.append(current)
                right.append(current)
            if w[current, label_i] > 0.5:
                subproblem_value = 1
            break
        elif branching:
            if main_grb_obj.X.at[i, selected_feature] == 1:  # going right on the branch
                left.append(current)
                target.append(current)
                current = main_grb_obj.tree.get_right_children(current)
            else:  # going left on the branch
                right.append(current)
                target.append(current)
                current = main_grb_obj.tree.get_left_children(current)

    return subproblem_value, left, right, target


##########################################################
# Defining the callback function
###########################################################
def benders_callback(model, where):
    """
    This function is called by Gurobi at every node through the branch-&-bound
    tree while we solve the model. Using the argument "where" we can see where
    the callback has been called. We are specifically interested at nodes
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
    X = model._main_grb_obj.X

    local_eps = 0.0001
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
                subproblem_value, left, right, target = subproblem(
                    model._main_grb_obj, b, p, w, i
                )
                if subproblem_value == 0:
                    added_cut = 1
                    lhs = get_cut_integer(
                        model._main_grb_obj, left, right, target, i
                    )
                    model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time