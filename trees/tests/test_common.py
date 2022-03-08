import pytest

from sklearn.utils.estimator_checks import check_estimator

from trees.StrongTree import StrongTreeClassifier
from trees.FairTree import FairTreeClassifier
from trees.RobustTree import RobustTreeClassifier
from trees.PrescriptiveTree import PrescriptiveTreeClassifier


@pytest.mark.skip()
@pytest.mark.parametrize(
    "estimator", [StrongTreeClassifier(depth=1, time_limit=1, _lambda=0.8)]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
