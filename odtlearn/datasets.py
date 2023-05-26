import numpy as np
import pandas as pd
import pkg_resources


def prescriptive_ex_data():
    """
    Return tuple of the train and test dataframes from the prescriptive tree example notebook
    """
    train_stream = pkg_resources.resource_stream(
        "odtlearn", "data/prescriptive/train_v2_100.csv"
    )
    test_stream = pkg_resources.resource_stream(
        "odtlearn", "data/prescriptive/test_v2_200.csv"
    )
    train = pd.read_csv(train_stream)
    test = pd.read_csv(test_stream)

    return train, test


def balance_scale_data():
    """
    Return a dataframe containing the balance-scale data set from the UCI ML repository.
    See the following URL for attribute information https://archive.ics.uci.edu/ml/datasets/Balance+Scale
    """
    stream = pkg_resources.resource_stream("odtlearn", "data/balance-scale_enc.csv")
    return pd.read_csv(stream)


def flow_oct_example():
    """
    Returns tuple with two numpy arrays containing the data used in the first example in the Flow OCTexample notebook in the ODTlearn documentation.
    The diagram within the code block shows the training dataset. Our dataset has
    two binary features (X1 and X2) and two class labels (+1 and -1).

    .. code-block:: text

        X2
        |               |
        |               |
        1    + +        |    -
        |               |
        |---------------|-------------
        |               |
        0    - - - -    |    + + +
        |    - - -      |
        |______0________|_______1_______X1

    Returns
    -------
    X: numpy array of covariates from training set
    y: numpy array of responses from training set
    """  # noqa: E501
    stream = pkg_resources.resource_stream(__name__, "data/example_1.npz")
    npzfile = np.load(stream)
    return npzfile["X"], npzfile["y"]


def robustness_example():
    """
    Returns tuple with three numpy arrays containing the data used in example 1
    of the RobustTree example notebook in the ODTlearn documentation.
    The diagram within the code block shows the training dataset. Our dataset has
    two binary features (X1 and X2) and two class labels (+1 and -1).

    .. code-block:: text

        X2
        |               |
        |               |
        1    + +        |    -
        |               |
        |---------------|-------------
        |               |
        0    - - - -    |    + + +
        |    - - -      |
        |______0________|_______1_______X1

    The third array returned contains a cost vector with the following form:
    - Uncertainty in 5 points at [0,0] on X1 can cause it to flip to [1,0] if needed to misclassify
    - Uncertainty in 1 point at [1,1] on X2 can cause it to flip to [1,0] if needed to misclassify
    - All other points certain

    Returns
    -------
    X: numpy array of covariates from training set
    y: numpy array of responses from training set
    costs: numpy array of costs for each observation in the training set
    """  # noqa: E501
    stream = pkg_resources.resource_stream("odtlearn", "data/example_1_robust.npz")
    npzfile = np.load(stream)
    return npzfile["X"], npzfile["y"], npzfile["costs"]


def example_2_data():
    """
    An example data set used to demonstrate usage of Flow OCT.
    The diagram within the code block shows the training dataset. Our dataset has
    two binary features (X1 and X2) and two class labels (+1 and -1). Here the data
    is imbalanced with the positive class being the minority class.

    .. code-block:: text

        X2
        |               |
        |               |
        1    + - -      |    -
        |               |
        |---------------|--------------
        |               |
        0    - - - +    |    - - -
        |    - - - -    |
        |______0________|_______1_______X1

    Returns
    -------
    X: numpy array of covariates from training set
    y: numpy array of responses from training set
    """  # noqa: E501
    stream = pkg_resources.resource_stream("odtlearn", "data/example_2.npz")
    npzfile = np.load(stream)
    return npzfile["X"], npzfile["y"]


def fairness_example():
    """
    A simulated data set used in the FairOCT example notebook.
    The diagram within the code block visualizes the training data.
    We have two binary features (X1, X2) and two class labels (+1 and -1).
    The protected feature is race and it has two levels (B and W).
    In the visualization of the training data, we see that, for example, there are 7 instances with (X1,X2) = (0,1) and among these 7 instances, 5 of them are from race W and 2 of them from race B. We also show the breakdown of the instances based on their class label.

    .. code-block:: text

        X2                    |
        |                     |
        1    5W: 4(-) 1(+)    |     2W: 1(-) 1(+)
        |    2B: 2(-)         |     5B: 3(-) 2(+)
        |                     |
        |                     |
        |---------------------|------------------------
        |                     |
        0    4W: 3(-) 1(+)    |         3W: 1(-) 2(+)
        |    1B:      1(+)    |         6B: 1(-) 5(+)
        |                     |
        |___________0_________|__________1_____________X1

    Returns
    -------
    X: numpy array of covariates from training set
    y: numpy array of responses from training set
    protect_feat: numpy array of the protected feature
    legit_factor: numpy array of the legitimate factor feature
    """  # noqa: E501

    stream = pkg_resources.resource_stream("odtlearn", "data/fairness_example.npz")
    npzfile = np.load(stream)
    return npzfile["X"], npzfile["y"], npzfile["protect_feat"], npzfile["legit_factor"]


def robust_example():
    """
    Return a dataframe containing the data set for MONK's second problem from the UCI ML repository.
    See the following URL for attribute information https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems
    """
    stream = pkg_resources.resource_stream("odtlearn", "data/monk2.csv")
    data = pd.read_csv(stream)
    y = data.pop("target")

    return data, y
