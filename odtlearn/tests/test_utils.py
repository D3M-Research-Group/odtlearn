import pandas as pd
import pytest
from numpy.testing import assert_allclose

from odtlearn.utils.StrongTreeUtils import binarize


@pytest.fixture
def example_data():
    number_of_child_list = [1, 2, 4, 3, 1, 2, 4, 3, 2, 1]
    age_list = [10, 20, 40, 30, 10, 20, 40, 30, 20, 10]
    race_list = [
        "Black",
        "White",
        "Hispanic",
        "Black",
        "White",
        "Black",
        "White",
        "Hispanic",
        "Black",
        "White",
    ]
    sex_list = ["M", "F", "M", "M", "F", "M", "F", "M", "M", "F"]
    df = pd.DataFrame(
        list(zip(sex_list, race_list, number_of_child_list, age_list)),
        columns=["sex", "race", "num_child", "age"],
    )
    return df


@pytest.fixture
def comparison_data():
    both_cols = pd.DataFrame.from_dict(
        {
            "sex_M": {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 1, 9: 0},
            "race_Black": {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1, 9: 0},
            "race_Hispanic": {
                0: 0,
                1: 0,
                2: 1,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 1,
                8: 0,
                9: 0,
            },
            "race_White": {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 0, 9: 1},
            "num_child_1": {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1},
            "num_child_2": {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0, 8: 1, 9: 1},
            "num_child_3": {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 1, 8: 1, 9: 1},
            "num_child_4": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
            "age_10": {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1},
            "age_20": {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0, 8: 1, 9: 1},
            "age_30": {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 1, 8: 1, 9: 1},
            "age_40": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
        }
    )
    cat_cols_only = pd.DataFrame.from_dict(
        {
            "sex_M": {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 1, 9: 0},
            "race_Black": {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1, 9: 0},
            "race_Hispanic": {
                0: 0,
                1: 0,
                2: 1,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 1,
                8: 0,
                9: 0,
            },
            "race_White": {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 0, 9: 1},
        }
    )
    int_cols_only = pd.DataFrame.from_dict(
        {
            "num_child_1": {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1},
            "num_child_2": {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0, 8: 1, 9: 1},
            "num_child_3": {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 1, 8: 1, 9: 1},
            "num_child_4": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
            "age_10": {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1},
            "age_20": {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0, 8: 1, 9: 1},
            "age_30": {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 1, 8: 1, 9: 1},
            "age_40": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
        }
    )
    return both_cols, cat_cols_only, int_cols_only


def test_binarize_need_lists(example_data):
    df = example_data
    with pytest.raises(
        AssertionError,
        match="Must provide at least one of the two options of a list"
        "of categorical columns or binary columns to binarize.",
    ):
        binarize(df, categorical_cols=[], integer_cols=[])


def test_binarize_correctness(example_data, comparison_data):
    df = example_data
    both_cols, cat_cols_only, int_cols_only = comparison_data
    assert_allclose(
        binarize(
            df, categorical_cols=["sex", "race"], integer_cols=["num_child", "age"]
        ),
        both_cols,
    )
    assert_allclose(
        binarize(df, categorical_cols=["sex", "race"], integer_cols=[]), cat_cols_only
    )

    assert_allclose(
        binarize(df, categorical_cols=[], integer_cols=["num_child", "age"]),
        int_cols_only,
    )
