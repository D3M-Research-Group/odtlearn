def pytest_addoption(parser):
    parser.addoption(
        "--test_gurobi",
        action="store_true",
        dest="test_gurobi",
        default=False,
        help="Enable Gurobi tests when a license is available",
    )


def pytest_configure(config):
    if not config.option.test_gurobi:
        setattr(config.option, "markexpr", "not test_gurobi")
