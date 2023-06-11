import numpy as np

from odtlearn.flow_oct import BendersOCT, FlowOCT
from odtlearn.utils.callbacks import benders_callback

# create data

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
y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])

######################################################################
# CBC BENDERSOCT TEST
######################################################################
clf_c = BendersOCT(solver="cbc", depth=1, time_limit=20, _lambda=0, verbose=True)

clf_c._extract_metadata(X, y)
clf_c._classes = [0, 1]
clf_c._create_main_problem()
# clf_c._solver.model.objective.expr
clf_c._solver.optimize(
    clf_c._g,
    clf_c._b,
    clf_c._p,
    clf_c._w,
    X=clf_c._X,
    obj=clf_c,
    solver=clf_c._solver,
    callback=True,
)

clf_c.b_value = clf_c._solver.get_var_value(clf_c._b, "b")
clf_c.w_value = clf_c._solver.get_var_value(clf_c._w, "w")
clf_c.p_value = clf_c._solver.get_var_value(clf_c._p, "p")


clf_g = BendersOCT(solver="gurobi", depth=1, time_limit=20, _lambda=0, verbose=False)

clf_g._extract_metadata(X, y)
clf_g._classes = [0, 1]
clf_g._create_main_problem()
clf_g._solver.model.params.Presolve = 0
clf_g._solver.optimize(callback=True, callback_action=benders_callback)

clf_g.b_value = clf_g._solver.get_var_value(clf_g._b, "b")
clf_g.w_value = clf_g._solver.get_var_value(clf_g._w, "w")
clf_g.p_value = clf_g._solver.get_var_value(clf_g._p, "p")


clf_c.p_value
clf_g.p_value

clf_g.b_value
clf_c.b_value

clf_g.w_value
clf_c.w_value


# ######################################################################
# # CBC FLOWOCT TEST
# ######################################################################
clf_c = FlowOCT(solver="cbc", depth=1, time_limit=20, _lambda=0, verbose=True)

clf_c._extract_metadata(X, y)
clf_c._classes = [0, 1]
clf_c._create_main_problem()
clf_c._solver.model.objective.expr
clf_c._solver.optimize()

clf_c.b_value = clf_c._solver.get_var_value(clf_c._b, "b")
clf_c.w_value = clf_c._solver.get_var_value(clf_c._w, "w")
clf_c.p_value = clf_c._solver.get_var_value(clf_c._p, "p")

clf_g = FlowOCT(solver="gurobi", depth=1, time_limit=20, _lambda=0, verbose=True)

clf_g._extract_metadata(X, y)
clf_g._classes = [0, 1]
clf_g._create_main_problem()
clf_g._solver.optimize()
clf_g.b_value = clf_g._solver.get_var_value(clf_g._b, "b")
clf_g.w_value = clf_g._solver.get_var_value(clf_g._w, "w")
clf_g.p_value = clf_g._solver.get_var_value(clf_g._p, "p")

clf_g.p_value
clf_c.p_value

clf_g.b_value
clf_c.b_value

clf_g.w_value
clf_c.w_value
