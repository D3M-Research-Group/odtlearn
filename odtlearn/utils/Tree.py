import numpy as np


class _Tree:
    """
    This class hold the classification/regression tree.
    We need following requirements:
        0- An easy way for user to create the tree
        1- The set of branching nodes
        2- The set of Leaves
        3- For each node we need to get the index of left/right children and the parent
    In this class we assume that we have a complete binary tree; we only receive the depth from the user
    """

    def __init__(self, d):
        self.depth = d
        self.Nodes = [i for i in range(1, np.power(2, d))]
        self.Leaves = [i for i in range(np.power(2, d), np.power(2, d + 1))]
        self.total_nodes = len(self.Nodes) + len(self.Leaves)

    def get_left_children(self, n):
        if n in self.Nodes:
            return 2 * n
        else:
            raise IndexError("Node index not found in tree")

    def get_right_children(self, n):
        if n in self.Nodes:
            return 2 * n + 1
        else:
            raise IndexError("Node index not found in tree")

    def get_parent(self, n):
        if (n in self.Nodes) or (n in self.Leaves):
            return np.floor(n / 2)
        else:
            raise IndexError("Node index not found in tree")

    def get_ancestors(self, n):
        ancestors = []
        if (n in self.Nodes) or (n in self.Leaves):
            current = n
            while current != 1:
                current = int(np.floor(current / 2))
                ancestors.append(current)
            return ancestors

        else:
            raise IndexError("Node index not found in tree")
