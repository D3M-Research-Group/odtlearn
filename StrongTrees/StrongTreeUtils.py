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

    if(isinstance(new_data, pd.Dataframe)):
        new_column_names = new_data.columns
        # take difference of sets
        non_matched_columns = set(new_column_names) - set(original_columns)
        if len(non_matched_columns) > 0:
            raise ValueError(
                f"Columns {list(non_matched_columns)} found in prediction data, but not found in fit data"
            )


def check_binary(df):
    # TO-DO: truncate output if lots of non_binary_columns
    non_binary_columns = [col for col in df
                          if not np.isin(df[col].dropna().unique(),
                                         [0, 1]).all()]
    if len(non_binary_columns) > 0:
        raise ValueError(
            f"Found columns ({non_binary_columns}) that contain values other than 0 or 1.")


def get_node_status(grb_model, b, beta, p, n):
    '''
    This function give the status of a given node in a tree. By status we mean whether the node
        1- is pruned? i.e., we have made a prediction at one of its ancestors
        2- is a branching node? If yes, what feature do we branch on
        3- is a leaf? If yes, what is the prediction at this node?
    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param beta: The values of prediction decision variable beta
    :param p: The values of decision variable p
    :param n: A valid node index in the tree
    :return: pruned, branching, selected_feature, leaf, value
    pruned=1 iff the node is pruned
    branching = 1 iff the node branches at some feature f
    selected_feature: The feature that the node branch on
    leaf = 1 iff node n is a leaf in the tree
    value: if node n is a leaf, value represent the prediction at this node
    '''
    tree = grb_model.tree
    mode = grb_model.mode
    pruned = False
    branching = False
    leaf = False
    value = None
    selected_feature = None

    p_sum = 0
    for m in tree.get_ancestors(n):
        p_sum = p_sum + p[m]
    if p[n] > 0.5:  # leaf
        leaf = True
        if mode == "regression":
            value = beta[n, 1]
        elif mode == "classification":
            for k in grb_model.labels:
                if beta[n, k] > 0.5:
                    value = k
    elif p_sum == 1:  # Pruned
        pruned = True

    if n in tree.Nodes:
        if (pruned is False) and (leaf is False):  # branching
            for f in grb_model.cat_features:
                if b[n, f] > 0.5:
                    selected_feature = f
                    branching = True

    return pruned, branching, selected_feature, leaf, value


def print_tree(grb_model, b, beta, p):
    '''
    This function print the derived tree with the branching features and the predictions asserted for each node
    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param beta: The values of prediction decision variable beta
    :param p: The values of decision variable p
    :return: print out the tree in the console
    '''
    tree = grb_model.tree
    for n in tree.Nodes + tree.Leaves:
        pruned, branching, selected_feature, leaf, value = get_node_status(
            grb_model, b, beta, p, n)
        print('#########node ', n)
        if pruned:
            print("pruned")
        elif branching:
            print(selected_feature)
        elif leaf:
            print('leaf {}'.format(value))


def get_predicted_value(grb_model, X, labels, b, beta, p):
    '''
    This function returns the predicted value for a given datapoint
    :param grb_model: The gurobi model we solved
    :param X: The dataset we want to compute accuracy for
    :param labels: A list of the column names for the X dataset
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :param i: Index of the datapoint we are interested in
    :return: The predicted value for datapoint i in dataset X
    '''
    tree = grb_model.tree
    predicted_values = np.array()
    for i in range(X.shape[0]):
        current = 1
        while True:
            pruned, branching, selected_feature, leaf, value = get_node_status(
                grb_model, b, beta, p, current)
            if leaf:
                predicted_values.append(value)
            elif branching:
                selected_feature_idx = np.where(labels == selected_feature)
                # Raise assertion error we don't have a column that matches
                # the selected feature or more than one column that matches
                assert len(selected_feature_idx) == 1, \
                    f"Found {len(selected_feature_idx)} columns matching the selected feature {selected_feature}"
                if X[i, selected_feature_idx] == 1:  # going right on the branch
                    current = tree.get_right_children(current)
                else:  # going left on the branch
                    current = tree.get_left_children(current)
    return predicted_values


def get_acc(grb_model, local_data, b, beta, p):
    '''
    This function returns the accuracy of the prediction for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The accuracy (fraction of datapoints which are correctly classified)
    '''
    label = grb_model.label
    acc = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, p, i)
        y_i = local_data.at[i, label]
        if yhat_i == y_i:
            acc += 1

    acc = acc / len(local_data.index)
    return acc


def get_mae(grb_model, local_data, b, beta, p):
    '''
    This function returns the MAE for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The MAE
    '''
    label = grb_model.label
    err = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, p, i)
        y_i = local_data.at[i, label]
        err += abs(yhat_i - y_i)

    err = err / len(local_data.index)
    return err


def get_mse(grb_model, local_data, b, beta, p):
    '''
    This function returns the MSE for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The MSE
    '''
    label = grb_model.label
    err = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, p, i)
        y_i = local_data.at[i, label]
        err += np.power(yhat_i - y_i, 2)

    err = err / len(local_data.index)
    return err


def get_r_squared(grb_model, local_data, b, beta, p):
    '''
    This function returns the R^2 for a given dataset
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :return: The R^2
    '''
    label = grb_model.label
    R_squared = 0
    y_bar = local_data[label].mean()
    # print(y_bar)
    SS_Residuals = 0
    SS_Total = 0
    for i in local_data.index:
        yhat_i = get_predicted_value(grb_model, local_data, b, beta, p, i)
        y_i = local_data.at[i, label]
        SS_Residuals += np.power(yhat_i - y_i, 2)
        SS_Total += np.power(y_bar - y_i, 2)

    R_squared = 1 - SS_Residuals / SS_Total
    return R_squared


def get_left_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f]
                   for f in master.cat_features if master.data.at[i, f] == 0)

    return lhs


def get_right_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f]
                   for f in master.cat_features if master.data.at[i, f] == 1)

    return lhs


def get_target_exp_integer(master, p, beta, n, i):
    label_i = master.data.at[i, master.label]

    if master.mode == "classification":
        lhs = -1 * master.beta[n, label_i]
    elif master.mode == "regression":
        # min (m[i]*p[n] - y[i]*p[n] + beta[n] , m[i]*p[n] + y[i]*p[n] - beta[n])
        if master.m[i] * p[n] - label_i * p[n] + beta[n, 1] < master.m[i] * p[n] + label_i * p[n] - beta[n, 1]:
            lhs = -1 * (master.m[i] * master.p[n] -
                        label_i * master.p[n] + master.beta[n, 1])
        else:
            lhs = -1 * (master.m[i] * master.p[n] +
                        label_i * master.p[n] - master.beta[n, 1])

    return lhs


def get_cut_integer(master, b, p, beta, left, right, target, i):
    lhs = LinExpr(0) + master.g[i]
    for n in left:
        tmp_lhs = get_left_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in right:
        tmp_lhs = get_right_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in target:
        tmp_lhs = get_target_exp_integer(master, p, beta, n, i)
        lhs = lhs + tmp_lhs

    return lhs


def subproblem(master, b, p, beta, i):
    label_i = master.data.at[i, master.label]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        pruned, branching, selected_feature, terminal, current_value = get_node_status(
            master, b, beta, p, current)
        if terminal:
            target.append(current)
            if current in master.tree.Nodes:
                left.append(current)
                right.append(current)
            if master.mode == "regression":
                subproblem_value = master.m[i] - abs(current_value - label_i)
            elif master.mode == "classification" and beta[current, label_i] > 0.5:
                subproblem_value = 1
            break
        elif branching:
            if master.data.at[i, selected_feature] == 1:  # going right on the branch
                left.append(current)
                target.append(current)
                current = master.tree.get_right_children(current)
            else:  # going left on the branch
                right.append(current)
                target.append(current)
                current = master.tree.get_left_children(current)

    return subproblem_value, left, right, target


##########################################################
# Defining the callback function
###########################################################
def benders_callback(model, where):
    '''
    This function is called by gurobi at every node through the branch-&-bound
    tree while we solve the model.Using the argument "where" we can see where
    the callback has been called. We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every datapoint we solve
    the subproblem which is a minimum cut and check if g[i] <= value of
    subproblem[i]. If this is violated we add the corresponding benders
    constraint as lazy constraint to the master problem and proceed.
    Whenever we have no violated constraint, it means that we have found
    the optimal solution.
    :param model: the gurobi model we are solving.
    :param where: the node where the callback function is called from
    :return:
    '''
    data_train = model._master.data
    mode = model._master.mode

    local_eps = 0.0001
    if where == GRB.Callback.MIPSOL:
        func_start_time = time.time()
        model._callback_counter_integer += 1
        # we need the value of b,w and g
        g = model.cbGetSolution(model._vars_g)
        b = model.cbGetSolution(model._vars_b)
        p = model.cbGetSolution(model._vars_p)
        beta = model.cbGetSolution(model._vars_beta)

        added_cut = 0
        # We only want indices that g_i is one!
        for i in data_train.index:
            if mode == "classification":
                g_threshold = 0.5
            elif mode == "regression":
                g_threshold = 0
            if g[i] > g_threshold:
                subproblem_value, left, right, target = subproblem(
                    model._master, b, p, beta, i)
                if mode == "classification" and subproblem_value == 0:
                    added_cut = 1
                    lhs = get_cut_integer(
                        model._master, b, p, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)
                elif mode == "regression" and ((subproblem_value + local_eps) < g[i]):
                    added_cut = 1
                    lhs = get_cut_integer(
                        model._master, b, p, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time
