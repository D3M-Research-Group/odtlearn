import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from odtlearn.StrongTree import StrongTreeClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.fixture
def synthetic_data_1():
    '''
    X2              |
    |               |
    1    + +        |    -
    |               |   
    |---------------|-------------
    |               |
    0    - - - -    |    + + +
    |    - - -      |
    |______0________|_______1_______X1
    '''
    X = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                 [1,0],[1,0],[1,0],
                 [1,1],
                 [0,1],[0,1]])
    y = np.array([0,0,0,0,0,0,0,1,1,1,0,1,1])

    return X, y


@pytest.fixture
def synthetic_data_2():
    '''
    X2              |
    |               |
    1    + - -      |    -
    |               |   
    |---------------|-------------
    |               |
    0    - - - +    |    - - -
    |    - - -      |
    |______0________|_______1_______X1
    '''
    X = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                  [1,0],[1,0],[1,0],
                  [1,1],
                  [0,1],[0,1],[0,1]])
    y = np.array([0,0,0,0,0,0,0,1,
                  0,0,0,
                  0,
                  1,0,0])

    return X, y

# Test that we raise a ValueError if X matrix has values other than zero or one
def test_StrongTree_X_nonbinary_error():

    clf = StrongTreeClassifier(depth=1, time_limit=2, _lambda=1)

    with pytest.raises(
        AssertionError,
        match="Expecting all values of covariate matrix to be either 0 or 1",
    ):
        X = np.arange(10).reshape(10, 1)
        y = np.random.randint(2, size=X.shape[0])
        clf.fit(X, y)

    with pytest.raises(
        ValueError,
        match=r"Found columns .* that contain values other than 0 or 1.",
    ):
        data = pd.DataFrame(
            {"x1": [1, 2, 2, 2, 3], "x2": [1, 2, 1, 0, 1], "y": [1, 1, -1, -1, -1]}
        )
        y = data.pop("y")
        clf.fit(data, y)


# Test that we raise an error if X and y have different number of rows
def test_StrongTree_X_data_shape_error():
    X = np.ones(100).reshape(100, 1)

    clf = StrongTreeClassifier(depth=1, time_limit=2, _lambda=1)

    with pytest.raises(
        ValueError, match="Found input variables with inconsistent numbers of samples"
    ):
        y_diff_size = np.random.randint(2, size=X.shape[0] + 1)
        clf.fit(X, y_diff_size)


@pytest.mark.test_gurobi
# Test that if we are given a pandas dataframe, we keep the original data and its labels
def test_StrongTree_classifier():
    train = pd.DataFrame(
        {"x1": [1, 0, 0, 0, 1], "x2": [1, 1, 1, 0, 1], "y": [1, 1, 0, 0, 0]},
        index=["A", "B", "C", "D", "E"],
    )
    y = train.pop("y")
    test = pd.DataFrame({"x1": [1, 1, 0, 0, 1], "x2": [1, 1, 1, 0, 1]})
    clf = StrongTreeClassifier(depth=1, time_limit=20, _lambda=0.2)

    clf.fit(train, y)
    # Test that after running the fit method we have b, w, and p
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")
    assert hasattr(clf, "b_value")
    assert hasattr(clf, "w_value")
    assert hasattr(clf, "p_value")

    y_pred = clf.predict(test)
    assert y_pred.shape == (train.shape[0],)







@pytest.mark.parametrize("d, l, benders, expected_pred", [(0, 0, False, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])), 
                                                          (1, 0, False, np.array([0,0,0,0,0,0,0,1,1,1,1,0,0])),
                                                          (2, 0, False, np.array([0,0,0,0,0,0,0,1,1,1,0,1,1])),
                                                          (0, 0, True, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])), 
                                                          (1, 0, True, np.array([0,0,0,0,0,0,0,1,1,1,1,0,0])),
                                                          (2, 0, True, np.array([0,0,0,0,0,0,0,1,1,1,0,1,1])),
                                                          (2, 0.51, False, np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])),
                                                          (2, 0.51, True, np.array([0,0,0,0,0,0,0,1,1,1,1,1,1]))])
def test_StrongTree_same_predictions(synthetic_data_1, d, l, benders, expected_pred):
    X, y = synthetic_data_1
    stcl = StrongTreeClassifier(
        depth = d, 
        time_limit = 100,
        _lambda = l,
        benders_oct= benders, 
        num_threads=None, 
        obj_mode = 'acc'
    )
    stcl.fit(X, y)
    stcl.print_tree()
    assert_allclose(stcl.predict(X), expected_pred)



@pytest.mark.parametrize("benders, obj_mode ,expected_pred", [(False, 'acc' , np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), 
                                                              (False, 'balance' ,np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1])),
                                                              (True, 'acc' , np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), 
                                                              (True, 'balance' ,np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]))])
def test_StrongTree_obj_mode(synthetic_data_2, benders, obj_mode, expected_pred):
    X, y = synthetic_data_2
    stcl = StrongTreeClassifier(
        depth = 2, 
        time_limit = 100,
        _lambda = 0,
        benders_oct= benders, 
        num_threads=None, 
        obj_mode = obj_mode
    )
    stcl.fit(X, y)
    stcl.print_tree()
    assert_allclose(stcl.predict(X), expected_pred)