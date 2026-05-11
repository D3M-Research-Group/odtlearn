import pytest

from odtlearn.solvers.gurobi_solver import GurobiSolver
from odtlearn.tests.test_utils import gurobi_available

# Gurobi Tests
def test_add_single_constraint_gb(skip_solver):
    if not gurobi_available():
        pytest.skip(reason="Gurobi not available")
    if skip_solver:
        pytest.skip(reason="Testing on github actions")
    gb_solver = GurobiSolver(False)
    gb_var = gb_solver.add_vars(1)
    gb_solver.add_constr(gb_var[0] == 1)
    gb_solver.model.update() # ensures that the model params are updated
    assert gb_solver.model.NumConstrs == 1

def test_set_objective_gb(skip_solver):
    if not gurobi_available():
        pytest.skip(reason="Gurobi not available")
    if skip_solver:
        pytest.skip(reason="Testing on github actions")

    gb_solver = GurobiSolver(False)
    gb_var = gb_solver.add_vars(1)
    obj_expression = gb_var[0] + 1
    with pytest.raises(TypeError, match="Objective sense must be integer or string."):
        gb_solver.set_objective(obj_expression, 2.0)

    with pytest.raises(ValueError, match="Invalid objective type: 2."):
        gb_solver.set_objective(obj_expression, 2)

    with pytest.raises(ValueError, match="Invalid objective type: SDFSDF."):
        gb_solver.set_objective(obj_expression, "SDFSDF")

#CBC Tests - skipped if Python MIP not installed (i.e., for Python 3.12 and up)
def test_add_single_constraint_cbc(skip_solver):
    pytest.importorskip("mip")
    if skip_solver:
        pytest.skip(reason="Testing on github actions")

    from odtlearn.solvers.cbc_solver import CBCSolver
    cbc_solver = CBCSolver(False)
    cbc_var = cbc_solver.add_vars(1)
    cbc_solver.add_constr(cbc_var[0] == 1)
    assert len(list(cbc_solver.model.constrs)) == 1

def test_prep_indices(skip_solver):
    pytest.importorskip("mip")
    if skip_solver:
        pytest.skip(reason="Testing on github actions")

    from odtlearn.solvers.cbc_solver import CBCSolver
    cbc_solver = CBCSolver(False)
    assert cbc_solver.prep_indices(1, [1.0, 2.0, 3.0], 2.0) == [
        [0],
        [1.0, 2.0, 3.0],
        [0, 1],
    ]

def test_set_objective_cbc(skip_solver):
    pytest.importorskip("mip")
    if skip_solver:
        pytest.skip(reason="Testing on github actions")

    from odtlearn.solvers.cbc_solver import CBCSolver
    cbc_solver = CBCSolver(False)
    cbc_var = cbc_solver.add_vars(1)
    obj_expression = cbc_var[0] + 1
    with pytest.raises(TypeError, match="Objective sense must be integer or string."):
        cbc_solver.set_objective(obj_expression, 2.0)

    with pytest.raises(ValueError, match="Invalid objective type: 2."):
        cbc_solver.set_objective(obj_expression, 2)

    with pytest.raises(ValueError, match="Invalid objective type: SDFSDF."):
        cbc_solver.set_objective(obj_expression, "SDFSDF")
