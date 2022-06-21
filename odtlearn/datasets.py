import pkg_resources
import pandas as pd
import numpy as np


def balance_scale_data():
    """Return a dataframe containing the balance-scale data set from the UCI ML repository.
    See the URL below for attribute information :https://archive.ics.uci.edu/ml/datasets/Balance+Scale
    """

    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream("odtlearn", "data/balance-scale_enc.csv")
    return pd.read_csv(stream)


def example_1_data():
    """Returns tuple with two numpy arrays containing the data used in example 1 of the example notebooks in the ODTlearn documentation.
    The diagram within the code block shows the training dataset. Our dataset has two binary features (X1 and X2) and two class labels (+1 and -1).

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

    """
    stream = pkg_resources.resource_stream("odtlearn", "data/example_1.npz")
    npzfile = np.load(stream)
    return npzfile["X"], npzfile["y"]


def example_2_data():
    """Returns tuple with two numpy arrays containing the data used in example 2 the example notebooks in the ODTlearn documentation.
    The diagram within the code block shows the training dataset. Our dataset has two binary features (X1 and X2) and two class labels (+1 and -1). Here the data is imbalanced with the positive class being the minority class.
    
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

    """
    stream = pkg_resources.resource_stream("odtlearn", "data/example_2.npz")
    npzfile = np.load(stream)
    return npzfile["X"], npzfile["y"]
