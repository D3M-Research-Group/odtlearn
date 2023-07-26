import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--test_solver",
        action="store_true",
        dest="test_solver",
        default=False,
        help="Enable solver tests when a license is available or we can easily install CBC binaries.",
    )


def pytest_configure(config):
    if not config.option.test_solver:
        setattr(config.option, "markexpr", "not test_solver")


@pytest.fixture
def skip_solver(request):
    return not request.config.getoption("--test_solver")
