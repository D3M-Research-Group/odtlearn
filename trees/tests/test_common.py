import pytest

from sklearn.utils.estimator_checks import check_estimator

from StrongTrees import StrongTreeEstimator
from StrongTrees import StrongTreeClassifier
from StrongTrees import StrongTreeTransformer


@pytest.mark.parametrize(
    "estimator",
    [StrongTreeEstimator(), StrongTreeTransformer(), StrongTreeClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
