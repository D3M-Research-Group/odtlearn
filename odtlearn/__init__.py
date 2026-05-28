from importlib.metadata import version

__version__ = version("odtlearn")


class ODTL:
    # optimization directions
    MIN = "MIN"
    MAX = "MAX"
    MINIMIZE = "MIN"
    MAXIMIZE = "MAX"

    # variable types
    BINARY = "B"
    CONTINUOUS = "C"
    INTEGER = "I"


name = "odtlearn"
