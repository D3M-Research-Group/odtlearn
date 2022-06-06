import numpy as np
import pandas as pd
import time
from gurobipy import LinExpr, quicksum, GRB
from sklearn.preprocessing import OneHotEncoder


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

    Parameters
    ----------
    grb_model :
        The gurobi model solved to optimality (or reached to the time limit).
    b :
        The values of branching decision variable b.
    w :
        The values of prediction decision variable w.
    p :
        The values of decision variable p
    n :
        A valid node index in the tree

    Returns
    -------
    pruned : int
        pruned=1 iff the node is pruned
    branching : int
        branching = 1 iff the node branches at some feature f
    selected_feature : str
        The feature that the node branch on
    leaf : int
        leaf = 1 iff node n is a leaf in the tree
    value :  double
        if node n is a leaf, value represent the prediction at this node
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


def print_tree_util(grb_model, b, w, p):
    """
    This function print the derived tree with the branching features and the predictions asserted for each node

    Parameters
    ----------
    grb_model :
        The gurobi model solved to optimality (or reached the time limit)
    b :
        The values of branching decision variable b
    w :
        The values of prediction decision variable w
    p :
        The values of decision variable p

    Returns
    -------
    Print out the tree in the console
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

    Parameters
    ----------
    grb_model :
        The gurobi model solved to optimality (or reached to the time limit)
    X :
        The dataset we want to compute accuracy for
    b :
        The value of decision variable b
    w :
        The value of decision variable w
    p :
        The value of decision variable p

    Returns
    -------
    predicted_values :
        The predicted value for dataset X
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
                selected_feature_idx = np.where(
                    grb_model.X_col_labels == selected_feature
                )
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
        -1 * main_grb_obj.b[n, f]
        for f in main_grb_obj.X_col_labels
        if main_grb_obj.X.at[i, f] == 0
    )

    return lhs


def get_right_exp_integer(main_grb_obj, n, i):
    lhs = quicksum(
        -1 * main_grb_obj.b[n, f]
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
    X = model._main_grb_obj.X

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
                    lhs = get_cut_integer(model._main_grb_obj, left, right, target, i)
                    model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time


def binarize(df, categorical_cols, integer_cols):
    """
    parameters
    ----------
    df: pandas dataframe
          A dataframe with only categorical/integer columns. There should not be any NA values.
    categorical_cols: list
                      a list consisting of the names of categorical columns of df
    integer_cols: list
                      a list consisting of the names of integer columns of df

    return
    ----------
    the binarized version of the input dataframe.

    This function encodes each categorical column as a one-hot vector, i.e.,
    for each level of the feature, it creates a new binary column with a value
    of one if and only if the original column has the corresponding level.
    A similar approach for encoding integer features is used with a slight change.
    The new binary column should have a value of one if and only if the main column
     has the corresponding value or any value smaller than it.
    """

    X_cat = np.array(df[categorical_cols])
    X_int = np.array(df[integer_cols])

    enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
    X_cat_enc = enc.fit_transform(X_cat).toarray()
    categorical_cols_enc = enc.get_feature_names_out(categorical_cols)

    enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
    X_int_enc = enc.fit_transform(X_int).toarray()
    integer_cols_enc = enc.get_feature_names_out(integer_cols)

    X_cat_enc = X_cat_enc.astype(int)
    X_int_enc = X_int_enc.astype(int)

    for col in integer_cols:
        col_enc_set = []
        col_offset = None
        for i, col_enc in enumerate(integer_cols_enc):
            if col in col_enc:
                col_enc_set.append(col_enc)
                if col_offset is None:
                    col_offset = i
        if len(col_enc_set) < 3:
            continue
        for i, col_enc in enumerate(col_enc_set):
            if i == 0:
                continue
            X_int_enc[:, (col_offset + i)] = (
                X_int_enc[:, (col_offset + i)] | X_int_enc[:, (col_offset + i - 1)]
            )

    df_enc = pd.DataFrame(
        np.c_[X_cat_enc, X_int_enc],
        columns=list(categorical_cols_enc) + list(integer_cols_enc),
    )
    return df_enc
