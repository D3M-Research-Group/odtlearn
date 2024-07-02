import pandas as pd
import pytest
from numpy.testing import assert_allclose

from odtlearn.utils.binarize import Binarizer


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
    cont_col = [
        0.60715055,
        0.86282283,
        0.93035626,
        0.85418288,
        0.9338212,
        0.38013132,
        0.36491731,
        0.72397201,
        0.70701631,
        0.08178343,
    ]
    df = pd.DataFrame(
        list(zip(sex_list, race_list, number_of_child_list, age_list, cont_col)),
        columns=["sex", "race", "num_child", "age", "cont_val"],
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
    cont_cols_only = pd.DataFrame.from_dict(
        {
            "cont_val_0": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 1,
            },
            "cont_val_1": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 1,
                7: 0,
                8: 0,
                9: 1,
            },
            "cont_val_2.0": {
                0: 1,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 1,
                7: 0,
                8: 1,
                9: 1,
            },
            "cont_val_3.0": {
                0: 1,
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 1,
            },
        }
    )

    return both_cols, cat_cols_only, int_cols_only, cont_cols_only


def test_binarizer_need_lists(example_data):
    df = example_data
    with pytest.raises(
        AssertionError,
        match="Must provide at least one of the three options of a list "
        "of categorical columns or integer columns or real valued columns to binarize.",
    ):
        Binarizer().fit(df)


def test_binarizer_correctness(example_data, comparison_data):
    df = example_data
    both_cols, cat_cols_only, int_cols_only, cont_cols_only = comparison_data

    binarizer = Binarizer(
        categorical_cols=["sex", "race"],
        integer_cols=["num_child", "age"],
        real_cols=[],
    )
    assert_allclose(
        binarizer.fit_transform(df),
        both_cols,
    )

    binarizer = Binarizer(
        categorical_cols=["sex", "race"], integer_cols=[], real_cols=[]
    )
    assert_allclose(
        binarizer.fit_transform(df),
        cat_cols_only,
    )

    binarizer = Binarizer(
        categorical_cols=[], integer_cols=["num_child", "age"], real_cols=[]
    )
    assert_allclose(
        binarizer.fit_transform(df),
        int_cols_only,
    )

    binarizer = Binarizer(
        categorical_cols=[], integer_cols=[], real_cols=["cont_val"], n_bins=4
    )
    assert_allclose(
        binarizer.fit_transform(df),
        cont_cols_only,
    )


def test_binarizer_feature_names(example_data):
    df = example_data
    binarizer = Binarizer(
        categorical_cols=["sex", "race"],
        integer_cols=["num_child", "age"],
        real_cols=["cont_val"],
        n_bins=3,
    )
    binarizer.fit(df)
    expected_features = [
        "sex_M",
        "race_Black",
        "race_Hispanic",
        "race_White",
        "num_child_1",
        "num_child_2",
        "num_child_3",
        "num_child_4",
        "age_10",
        "age_20",
        "age_30",
        "age_40",
        "cont_val_0",
        "cont_val_1",
        "cont_val_2",
    ]
    assert set(binarizer.column_names_) == set(expected_features)


def test_binarizer_transform(example_data):
    df = example_data
    binarizer = Binarizer(
        categorical_cols=["sex", "race"],
        integer_cols=["num_child", "age"],
        real_cols=["cont_val"],
        n_bins=3,
    )
    X_bin = binarizer.fit_transform(df)
    assert X_bin.shape == (10, 15)
    assert all(X_bin.dtypes == "float64")
