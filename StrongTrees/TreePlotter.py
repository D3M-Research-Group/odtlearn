#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt


# tree_data = {'top node': {0: 'no', 1: {'next node': {0: 'no',
#  1: {'idk': {'a': 'yes', 'b': 'no'}}}
#                                            }
#                               },
#              }


class TreePlotter:
    """ A class for plotting a decision tree using matplotlib.

    The class parses a decision tree in the form of a nested dictionary
    and draws the nodes on the figure

    Parameters
    ----------
    Tree : dict
        Decision tree in the form of a nested dictionary

    Attributes
    ----------
    decisionNode: dict
        dictionary containing styling parameters for the decision
        nodes which are used when we add the node to the figure in `plotNode`

    leafNode: dict
        dictionary containing styling parameters for the leaf
        nodes which are used when we add the node to the figure in `plotNode`

    arrow_args: dict
        dictionary containing styling parameters for the arrows connecting
        the nodes of the tree. `shrinkB` is set to 15 by default to
        prevent the arrows from being drawn over the node text

    """

    def __init__(self, Tree):
        # plot styling parameters
        self.decisionNode = dict(boxstyle="round", fc="0.8")
        self.leafNode = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-", shrinkB=15)

        # initialize the figure itself, hide plot tick marks
        self.fig = plt.figure(1, facecolor='white')
        self.axprops = dict(xticks=[], yticks=[])
        self.ax = self.fig.add_subplot(111, frameon=False, **self.axprops)

        # get information about the tree
        self.Tree = Tree
        self.totalW = float(self.get_num_leaves(self.Tree))
        self.totalD = float(self.get_tree_depth(self.Tree))

        # define spacing between the nodes between levels and within levels
        self.xOff = -0.5/self.totalW
        self.yOff = 1.0

    # num_leaves function

    def get_num_leaves(self, Tree):
        num_leaves = 0
        first_list = list(Tree.keys())
        first_str = first_list[0]
        second_dict = Tree[first_str]
        for key in second_dict.keys():
            if isinstance(second_dict[key], dict):
                num_leaves += self.get_num_leaves(second_dict[key])
            else:
                num_leaves += 1
        return num_leaves

    # depths function

    def get_tree_depth(self, Tree):
        max_depth = 0
        first_list = list(Tree.keys())
        first_str = first_list[0]
        second_dict = Tree[first_str]
        for key in second_dict.keys():
            if isinstance(second_dict[key], dict):
                this_depth = 1 + self.get_tree_depth(second_dict[key])
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
        return max_depth

    def plotNode(self, nodeText, centerPt, parentPt, nodeType):
        self.ax.annotate(nodeText,
                         xy=parentPt,
                         xycoords='axes fraction',
                         xytext=centerPt,
                         textcoords='axes fraction',
                         va='center',
                         ha='center',
                         bbox=nodeType,
                         arrowprops=self.arrow_args)

    # define the main functions, plotTree
    def plotTree(self, Tree, parentPt, nodeTxt):
        # if the first key tells you what feat was split on
        # this determines the x width of this tree
        num_leaves = self.totalW
        first_list = list(Tree.keys())
        # the text label for this node should be this
        first_str = first_list[0]
        cntrPt = (self.xOff + (1.0 + float(num_leaves)) /
                  2.0/self.totalW, self.yOff)
        self.plotNode(first_str, cntrPt, parentPt, self.decisionNode)
        secondDict = Tree[first_str]
        self.yOff = self.yOff - 1.0/self.totalD
        for key in secondDict.keys():
            # test to see if the nodes are dictonaires,
            # if not they are leaf nodes
            if isinstance(secondDict[key], dict):
                self.plotTree(secondDict[key], cntrPt, str(key))  # recurse
            else:  # it's a leaf node print the leaf node
                self.xOff = self.xOff +\
                    1.0/self.totalW
                self.plotNode(secondDict[key], (self.xOff,
                                                self.yOff),
                              cntrPt, self.leafNode)
        self.yOff = self.yOff + 1.0/self.totalD
        return self.fig

    def plot(self):
        self.plotTree(self.Tree, (0.5, 1.0), '')
