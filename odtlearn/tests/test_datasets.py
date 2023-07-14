from odtlearn import datasets


def test_load_balance_scale_data():
    data_tuple = datasets.balance_scale_data()
    assert len(data_tuple) == 625


def test_load_prescriptive_data():
    data_tuple = datasets.prescriptive_ex_data()
    assert len(data_tuple) == 2


def test_load_flow_oct_data():
    data_tuple = datasets.flow_oct_example()
    assert len(data_tuple) == 2


def test_load_robustness_example_data():
    data_tuple = datasets.robustness_example()
    assert len(data_tuple) == 3


def test_load_example_2_data():
    data_tuple = datasets.example_2_data()
    assert len(data_tuple) == 2


def test_load_fairness_example():
    data_tuple = datasets.fairness_example()
    assert len(data_tuple) == 4


def test_load_robust_example():
    data_tuple = datasets.robust_example()
    assert len(data_tuple) == 2
