import numpy as np


def print_tree(grb_model, b, w, p):
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
        for k in grb_model.treatments_set:
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
