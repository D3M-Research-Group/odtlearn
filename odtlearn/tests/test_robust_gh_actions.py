import numpy as np
import pandas as pd

from odtlearn.robust_oct import RobustOCT

X = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 1],
    ]
)
X = pd.DataFrame(X, columns=["X1", "X2"])


y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])
robust_classifier = RobustOCT(solver="cbc", depth=2, time_limit=60)
robust_classifier.fit(X, y)
predictions = robust_classifier.predict(X)


# Note: 10 is a proxy for infinite cost, as it is over the allowed budgets we will specify
costs = np.array(
    [
        [1, 10],
        [1, 10],
        [1, 10],
        [1, 10],
        [1, 10],
        [10, 10],
        [10, 10],
        [10, 10],
        [10, 10],
        [10, 10],
        [10, 1],
        [10, 10],
        [10, 10],
    ]
)
costs = pd.DataFrame(costs, columns=["X1", "X2"])


robust_classifier = RobustOCT(
    solver="cbc",
    depth=2,
    time_limit=60,
)
robust_classifier.fit(X, y, costs=costs, budget=2)
predictions = robust_classifier.predict(X)
