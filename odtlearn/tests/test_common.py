import pytest

from sklearn.utils.estimator_checks import check_estimator

from odtlearn.StrongTree import StrongTreeClassifier
from odtlearn.FairTree import FairTreeClassifier
from odtlearn.RobustTree import RobustTreeClassifier
from odtlearn.PrescriptiveTree import PrescriptiveTreeClassifier


@pytest.mark.skip()
@pytest.mark.parametrize(
    "estimator", [StrongTreeClassifier(depth=1, time_limit=1, _lambda=0.8)]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
