import pytest

from sklearn.utils.estimator_checks import check_estimator

from StrongTrees import TemplateEstimator
from StrongTrees import TemplateClassifier
from StrongTrees import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
