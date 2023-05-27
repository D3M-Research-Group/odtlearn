import pytest

from odtlearn.utils.Tree import _Tree


def test_tree_exceptions():
    tree = _Tree(4)

    # Try to get left child that does not exist
    with pytest.raises(IndexError, match="Node index not found in tree"):
        tree.get_left_children(17)

    # Try to get right child that does not exist
    with pytest.raises(IndexError, match="Node index not found in tree"):
        tree.get_right_children(16)

    # Try to get parent that DNE
    with pytest.raises(IndexError, match="Node index not found in tree"):
        tree.get_parent(0)

    # Try to get ancestors that don't exist
    with pytest.raises(IndexError, match="Node index not found in tree"):
        tree.get_ancestors(0)


def test_tree_size():
    tree = _Tree(3)

    assert tree.total_nodes == 15
    assert tree.Nodes == list(range(1, 8))
    assert tree.Leaves == list(range(8, 16))
