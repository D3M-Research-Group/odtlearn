import pytest

from odtlearn.utils.solver import Solver


def test_callback_action(skip_solver):

    if skip_solver:
        pytest.skip(reason="No gurobi license available.")

    gb_solver = Solver(solver_name="gurobi", verbose=False)
    with pytest.raises(
        ValueError, match="Must supply callback action if callback=True"
    ):
        gb_solver.optimize(None, None, None, callback=True, callback_action=None)

    cbc_solver = Solver(solver_name="cbc", verbose=False)

    with pytest.raises(
        ValueError, match="Must supply callback action if callback=True"
    ):
        cbc_solver.optimize(None, None, None, callback=True, callback_action=None)


def test_add_single_constraint(skip_solver):
    if skip_solver:
        pytest.skip(reason="No gurobi license available.")
    gb_solver = Solver(solver_name="gurobi", verbose=False)
    gb_var = gb_solver.add_vars(1)
    gb_solver.add_constr(gb_var[0] == 1)
    assert len(list(gb_solver.model.constrs)) == 1

    cbc_solver = Solver(solver_name="cbc", verbose=False)
    cbc_var = cbc_solver.add_vars(1)
    cbc_solver.add_constr(cbc_var[0] == 1)
    assert len(list(cbc_solver.model.constrs)) == 1


def test_prep_indices(skip_solver):
    if skip_solver:
        pytest.skip(reason="No gurobi license available.")
    cbc_solver = Solver(solver_name="cbc", verbose=False)
    assert cbc_solver.prep_indices(1, [1.0, 2.0, 3.0], 2.0) == [
        [0],
        [1.0, 2.0, 3.0],
        [0, 1],
    ]


def test_set_objective(skip_solver):
    if skip_solver:
        pytest.skip(reason="No gurobi license available.")

    cbc_solver = Solver(solver_name="cbc", verbose=False)
    cbc_var = cbc_solver.add_vars(1)
    obj_expression = cbc_var[0] + 1
    with pytest.raises(TypeError, match="Objective sense must be integer or string."):
        cbc_solver.set_objective(obj_expression, 2.0)

    with pytest.raises(ValueError, match="Invalid objective type: 2."):
        cbc_solver.set_objective(obj_expression, 2)

    with pytest.raises(ValueError, match="Invalid objective type: SDFSDF."):
        cbc_solver.set_objective(obj_expression, "SDFSDF")
