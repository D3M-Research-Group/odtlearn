from importlib.metadata import version

__version__ = version("odtlearn")
name = "odtlearn"
"""Name of the ODTLearn module."""


class ODTL:
    """
    Top-level constants used for ODTLearn solvers.

    Attributes
    ----------
    MIN : str
        Minimization problem indicator, equivalent to ODTL.MINIMIZE
    MAX : str
        Maximization problem indicator, equivalent to ODTL.MAXIMIZE
    MINIMIZE : str
        Minimization problem indicator, equivalent to ODTL.MIN
    MAXIMIZE : str
        Maximization problem indicator, equivalent to ODTL.MAX
    BINARY : str
        Binary variable indicator
    CONTINUOUS : str
        Continuous variable indicator
    INTEGER : str
        Integer variable indicator
    """

    # optimization directions
    MIN = "MIN"
    MAX = "MAX"
    MINIMIZE = "MIN"
    MAXIMIZE = "MAX"

    # variable types
    BINARY = "B"
    CONTINUOUS = "C"
    INTEGER = "I"
