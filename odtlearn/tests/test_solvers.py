import pytest

from odtlearn.utils.cbc_solver import CBCSolver
from odtlearn.utils.gb_solver import GurobiSolver


def test_callback_action():
    gb_solver = GurobiSolver()
    cbc_solver = CBCSolver(verbose=False)
    with pytest.raises(
        ValueError, match="Must supply callback action if callback=True"
    ):
        gb_solver.optimize(None, None, None, callback=True, callback_action=None)

    with pytest.raises(
        ValueError, match="Must supply callback action if callback=True"
    ):
        cbc_solver.optimize(None, None, None, callback=True, callback_action=None)


def test_add_single_constraint():
    gb_solver = GurobiSolver()
    cbc_solver = CBCSolver(verbose=False)

    gb_var = gb_solver.add_vars(1)
    gb_solver.add_constr(gb_var[0] == 1)
    gb_solver.model.update()
    assert len(gb_solver.model.getConstrs()) == 1

    cbc_var = cbc_solver.add_vars(1)
    cbc_solver.add_constr(cbc_var[0] == 1)
    assert len(list(cbc_solver.model.constrs)) == 1


def test_prep_indices():
    cbc_solver = CBCSolver(verbose=False)
    assert cbc_solver.prep_indices(1, [1.0, 2.0, 3.0], 2.0) == [
        [0],
        [1.0, 2.0, 3.0],
        [0, 1],
    ]


def test_set_objective():
    cbc_solver = CBCSolver(verbose=False)
    cbc_var = cbc_solver.add_vars(1)
    obj_expression = cbc_var[0] + 1
    with pytest.raises(TypeError, match="Objective sense must be integer or string."):
        cbc_solver.set_objective(obj_expression, 2.0)

    with pytest.raises(ValueError, match="Invalid objective type: 2."):
        cbc_solver.set_objective(obj_expression, 2)

    with pytest.raises(ValueError, match="Invalid objective type: SDFSDF."):
        cbc_solver.set_objective(obj_expression, "SDFSDF")
