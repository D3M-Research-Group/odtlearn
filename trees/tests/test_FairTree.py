import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from trees.FairTree import FairTreeClassifier
from trees.StrongTree import StrongTreeClassifier
from trees.utils.StrongTreeUtils import print_tree 


# @pytest.fixture
# def data():
#     return load_iris(return_X_y=True)

@pytest.fixture
def synthetic_data_1():
    '''
     This is the data we generate in this function
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
    '''
    X = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],
                    [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
                    [1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],
                    [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
    P = np.array([0,0,0,0,1,
                    0,0,0,1,1,1,1,1,1,
                    0,0,1,1,1,1,1,
                    0,0,0,0,0,1,1])
    y = np.array([0,0,0,1,1,
                    0,1,1,0,1,1,1,1,1,
                    0,1,0,0,0,1,1,
                    0,0,0,0,1,0,0])

    P = P.reshape(-1,1)
    # X = np.concatenate((X, P), axis=1)
    l = X[:,1]
    
    return X, y, P, l

# def test_FairTree_transformer_error(data):
#     X, y = data
#     trans = FairTreeClassifier()
#     trans.fit(X)
#     with pytest.raises(ValueError, match="Shape of input is different"):
#         X_diff_size = np.ones((10, X.shape[1] + 1))
#         trans.transform(X_diff_size)


# def test_FairTree_transformer(data):
#     X, y = data
#     trans = FairTreeTransformer()
#     assert trans.demo_param == "demo"

#     trans.fit(X)
#     assert trans.n_features_ == X.shape[1]

#     X_trans = trans.transform(X)
#     assert_allclose(X_trans, np.sqrt(X))

#     X_trans = trans.fit_transform(X)
#     assert_allclose(X_trans, np.sqrt(X))


# def test_FairTree_classifier(data):
#     X, y = data
#     clf = FairTreeClassifier()
#     assert clf.demo_param == "demo"

#     clf.fit(X, y)
#     assert hasattr(clf, "classes_")
#     assert hasattr(clf, "X_")
#     assert hasattr(clf, "y_")

#     y_pred = clf.predict(X)
#     assert y_pred.shape == (X.shape[0],)


def test_FairTree_same_predictions(synthetic_data_1):
    X, y, P, l = synthetic_data_1
    fcl = FairTreeClassifier(
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_type=None,
        fairness_bound=1,
        num_threads=None,
        obj_mode = 'acc'
    ) 

    stcl = StrongTreeClassifier(
        depth = 2, 
        time_limit = 100,
        _lambda = 0,
        benders_oct= False, 
        num_threads=None, 
        obj_mode = 'acc'
    )

    stcl.fit(X, y) 
    stcl_pred = stcl.predict(X)

    fcl.fit(X, y, P, l)
    fcl_pred = fcl.predict(X)
    
    assert_allclose(fcl_pred, stcl_pred)


@pytest.mark.parametrize("f, b, g0_value", [('SP', 1, 0.214),
                                         ('SP', 0.2, 0.5),
                                         ('PE', 1, 0.111),
                                         ('PE', 0.04, 0)])
def test_FairTree_metrics(synthetic_data_1, f, b, g0_value):
    X, y, P, l = synthetic_data_1
    fcl = FairTreeClassifier(
        positive_class=1,
        depth=2,
        _lambda=0,
        time_limit=100,
        fairness_type=f,
        fairness_bound=b,
        num_threads=None,
        obj_mode = 'acc'
    )   

    fcl.fit(X, y, P, l)
    if f=='SP':
        sp_val = fcl.get_SP(P, fcl.predict(X))
        assert_allclose(np.round(sp_val[(0,1)],3), g0_value)
    elif f=='PE':
        eq_val = fcl.get_EqOdds(P, y, fcl.predict(X))
        assert_allclose(np.round(eq_val[(0,0,1)],3), g0_value)