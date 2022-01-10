import pytest

from sklearn.utils.estimator_checks import check_estimator

from FairTrees import FairTreeEstimator
from FairTrees import FairTreeClassifier
from FairTrees import FairTreeTransformer


@pytest.mark.parametrize(
    "estimator",
    [FairTreeEstimator(), FairTreeTransformer(), FairTreeClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
