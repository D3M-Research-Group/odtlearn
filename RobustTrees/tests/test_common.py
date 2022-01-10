import pytest

from sklearn.utils.estimator_checks import check_estimator

from RobustTrees import RobustTreeEstimator
from RobustTrees import RobustTreeClassifier
from RobustTrees import RobustTreeTransformer


@pytest.mark.parametrize(
    "estimator",
    [RobustTreeEstimator(), RobustTreeTransformer(), RobustTreeClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
