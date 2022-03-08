from collections import defaultdict
from functools import reduce, partial
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
from odtlearn.utils.StrongTreeUtils import get_node_status


def contain_dict(list_var):
    res = False
    for v in list_var:
        if isinstance(v, (dict, defaultdict)):
            res = True
    return res


def get_dict_from_list(tree_dict, key_list, counter=0):
    key = key_list[counter]
    next_dict = tree_dict.get(key, None)
    if next_dict is not None:
        get_dict_from_list(next_dict, key_list[counter + 1 :], counter + 1)


def dictFromList(dict_list, mapListItem, mapList, nextItem):
    print(f"dict_list: {dict_list}")
    print(f"mapListItem: {mapListItem}")
    print(f"nextItem: {nextItem}")
    if isinstance(dict_list[mapListItem], list):
        print(f"dict_list[mapListItem]: {dict_list[mapListItem]}")
        # need to loop through the items in the list at that level
        # if they are dictionaries, we need to check that the key is in the names list
        types = [isinstance(item, dict) for item in dict_list[mapListItem]]
        if sum(types) > 1:
            for item in dict_list[mapListItem]:
                print(f"item: {item}")
                print(f"string: {list(item.keys())[0]}")
                if list(item.keys())[0] in mapList or list(item.keys())[0] == nextItem:
                    return item
        else:
            # otherwise there is only one branch and one leaf
            if isinstance(dict_list[mapListItem][0], dict):
                return dict_list[mapListItem][0]
            else:
                return dict_list[mapListItem][1]
    else:
        return dict_list[mapListItem]


def getFromDict(dataDict, mapList, nextItem):
    # need to be able to pass the list of items each time
    print(partial(dictFromList, mapList=mapList, nextItem=nextItem))
    return reduce(
        partial(dictFromList, mapList=mapList, nextItem=nextItem), mapList, dataDict
    )


def setInDict(dataDict, mapList, value, leaf=False):
    if leaf:
        getFromDict(dataDict, mapList[:-1], mapList[-1])[mapList[-1]].append(value)
    else:
        getFromDict(dataDict, mapList[:-1], mapList[-1])[mapList[-1]].append(
            {value: []}
        )


def ancestor_names(ancestors, b, column_names):
    names = []
    for idx in ancestors:
        for f in column_names:
            if b[idx, f] > 0.5:
                names.append(f)
    return names


class Node:
    def __init__(
        self,
        text="",
        x=None,
        y=None,
        isRoot=False,
        parentNode=None,
        leftNode=None,
        rightNode=None,
        textBox=None,
        boxWidth=None,
    ):
        self.x = x
        self.y = y
        self.text = text
        self.IsRoot = isRoot
        self.parentNode = parentNode
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.textBox = textBox
        self.boxWidth = boxWidth
        self.isTerminal = False

    def getParentNode(self):
        return self.parentNode

    def setParentNode(self, parentNode):
        self.parentNode = parentNode

    def getIsTerminal(self):
        return self.isTerminal

    def setIsTerminal(self, isTerminal):
        self.isTerminal = isTerminal
        return

    def getLeftNode(self):
        return self.leftNode

    def setLeftNode(self, leftNode):
        self.leftNode = leftNode

    def getRightNode(self):
        return self.rightNode

    def setRightNode(self, rightNode):
        self.rightNode = rightNode

    def getText(self):
        return self.text

    def setText(self, text):
        self.text = text

    def getX(self):
        return self.x

    def setX(self, x):
        self.x = x

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y

    def __str__(self):
        return f"{self.x}, {self.y}, {self.text}"

    def __repr__(self):
        return f"TreeNode({self.x}, {self.y}, {self.text})"


class Tree:
    def __init__(self):
        self.depth = 0
        self.width = 1
        self.height = 1
        self.verticalSpace = 0.5
        self.xMax = -np.infty
        self.xMin = np.infty
        self.yMax = -np.infty
        self.yMin = np.infty

    def createNode(
        self,
        text="",
        x=None,
        y=None,
        isRoot=False,
        parentNode=None,
        leftNode=None,
        rightNode=None,
        isTerminal=False,
    ):
        return Node(text, x, y, isRoot, parentNode, leftNode, rightNode, isTerminal)

    def addNode(
        self,
        node=None,
        isLeft=True,
        text="",
        x=None,
        y=None,
        isRoot=False,
        parentNode=None,
        isTerminal=False,
    ):
        if x + self.width + 2 > self.xMax:
            self.setXMax(x + self.width)
        if x - self.width < self.xMin:
            self.setXMin(x - self.width)
        if y > self.yMax:
            self.setYMax(y)
        if y - self.height - self.verticalSpace < self.yMin:
            self.setYMin(y - self.height)

        if node is None:
            return self.createNode(text, x, y, isRoot=True)
        assert isinstance(node, Node)
        if isLeft:
            node.leftNode = self.createNode(
                text, x, y, parentNode=node, isTerminal=isTerminal
            )
            return node.leftNode
        else:
            node.rightNode = self.createNode(
                text, x, y, parentNode=node, isTerminal=isTerminal
            )
            return node.rightNode

    def getXMax(self):
        return self.xMax

    def setXMax(self, xMax):
        self.xMax = xMax
        return

    def getXMin(self):
        return self.xMin

    def setXMin(self, xMin):
        self.xMin = xMin
        return

    def getYMax(self):
        return self.yMax

    def setYMax(self, yMax):
        self.yMax = yMax
        return

    def getYMin(self):
        return self.yMin

    def setYMin(self, yMin):
        self.yMin = yMin
        return


class TreePlotter:
    def __init__(self, tree, labels, column_names, b, w, p) -> None:
        self.tree = tree
        self.labels = labels
        self.column_names = column_names
        self.b = b
        self.w = w
        self.p = p
        self.nested_tree = Tree()
        self.root = self.nested_tree.addNode(x=0, y=0, text="root")

    def make_nested_dict(self):
        self.tree_dict = {}
        for index in self.tree.Nodes + self.tree.Leaves:
            # get information about current node
            pruned, branching, selected_feature, leaf, value = get_node_status(
                self.tree, self.labels, self.column_names, self.b, self.w, self.p, index
            )
            if not pruned:
                if branching:
                    ancestors = self.tree.get_ancestors(index)[::-1]
                    if len(ancestors) > 0:
                        # get the name of those ancesters
                        names = ancestor_names(ancestors, self.b, self.column_names)
                        setInDict(self.tree_dict, names, selected_feature)
                    else:
                        self.tree_dict[selected_feature] = []
                elif leaf:
                    ancestors = self.tree.get_ancestors(index)[::-1]
                    names = ancestor_names(ancestors, self.b, self.column_names)
                    setInDict(self.tree_dict, names, str(value), True)

    def building_tree_node(
        self, tree_dict, tree_class, node, counter=0, left_branch=True
    ):
        box_x = 2
        counter += 1
        for key, value in tree_dict.items():
            print(counter)
            print(key)
            print(value)
            node.setText(key)
            if contain_dict(value):
                if isinstance(value[0], dict):
                    # print(list(value[0].keys())[0])
                    left_node = tree_class.addNode(
                        node=node,
                        x=-box_x * counter,
                        y=-1 * counter,
                        isLeft=True,
                        text=list(value[0].keys())[0],
                    )
                    print(left_node)
                    print(
                        f"entering recursion following left node with level={counter}"
                    )
                    self.building_tree_node(
                        tree_dict=value[0],
                        tree_class=tree_class,
                        node=left_node,
                        counter=counter,
                    )
                    print(
                        f"exited recursion following left node, returning to level={counter}"
                    )
                    if isinstance(value[1], dict):
                        right_node = tree_class.addNode(
                            node=node,
                            x=box_x * counter,
                            y=-1 * counter,
                            isLeft=False,
                            text=list(value[1].keys())[0],
                        )
                        print(right_node)
                        print(
                            f"entering recursion following right node with level={counter}"
                        )
                        self.building_tree_node(
                            tree_dict=value[1],
                            tree_class=tree_class,
                            node=right_node,
                            counter=counter,
                            left_branch=False,
                        )
                        print(
                            f"exited recursion following right node, returning to level={counter}"
                        )
                    else:
                        temp = tree_class.addNode(
                            node=node,
                            x=box_x * counter,
                            y=-1 * counter,
                            isLeft=False,
                            text=value[1],
                        )
                        temp.setIsTerminal(True)
                        print(temp)
                        print(f"reached leaf at level={counter}")
                elif isinstance(value[1], dict):
                    right_node = tree_class.addNode(
                        node=node,
                        x=box_x * counter,
                        y=-1 * counter,
                        isLeft=False,
                        text=list(value[1].keys())[0],
                    )
                    print(right_node)
                    print(
                        f"entering recursion following right node with level={counter}"
                    )
                    self.building_tree_node(
                        tree_dict=value[1],
                        tree_class=tree_class,
                        node=right_node,
                        counter=counter,
                        left_branch=False,
                    )
                    print(
                        f"exited recursion following right node, returning to level={counter}"
                    )
                    temp = tree_class.addNode(
                        node=node,
                        x=-box_x * counter,
                        y=-1 * counter,
                        isLeft=True,
                        text=value[0],
                    )
                    temp.setIsTerminal(True)
                    print(temp)
                    print(f"reached leaf at level={counter}")
            else:
                temp = tree_class.addNode(
                    node=node,
                    x=-box_x * counter if left_branch else box_x * counter,
                    y=-1 * counter,
                    isLeft=True,
                    text=value[0],
                )
                temp.setIsTerminal(True)

                print(f"reached leaves at level={counter}")
                print(temp)
                temp = tree_class.addNode(
                    node=node,
                    x=-(box_x * counter) / 2 + 2
                    if left_branch
                    else (box_x * counter) / 2 - 2,
                    y=-1 * counter,
                    isLeft=False,
                    text=value[1],
                )
                temp.setIsTerminal(True)
                print(temp)
        print(f"finished at level={counter}, returning now")
        # self.final_tree = tree_class
        # self.final_root = node
        return tree_class, node

    def drawNode(self, node, renderer):
        if node is not None:
            if node.getIsTerminal():
                bbox = dict(boxstyle="square", fc="green")
            else:
                bbox = dict(boxstyle="square", fc="yellow")
            text_box = self.ax.text(
                node.getX(),
                node.getY(),
                node.getText(),
                bbox=bbox,
                fontsize=15,
                ha="center",
                va="center",
            )
            node.textBox = text_box
            if node.parentNode is not None:
                parentTextBox = node.parentNode.textBox
                transf = self.ax.transData.inverted()
                pbb = parentTextBox.get_window_extent(renderer=renderer)
                pbb_datacoords = pbb.transformed(transf)
                self.ax.plot(
                    (node.parentNode.x, node.x),
                    (node.parentNode.y - pbb_datacoords.height * 0.7, node.y),
                    color="k",
                )
            self.drawNode(node.leftNode, renderer=renderer)
            self.drawNode(node.rightNode, renderer=renderer)

    def plot(self):
        # TO-DO: add kwargs for passing to plt.figure
        self.make_nested_dict()
        self.final_tree, self.final_root = self.building_tree_node(
            self.tree_dict, self.nested_tree, self.root
        )
        self.fig = plt.figure(figsize=(10, 10))
        renderer = self.fig.canvas.get_renderer()
        self.ax = self.fig.add_subplot()

        self.ax.set_xlim(self.final_tree.getXMin(), self.final_tree.getXMax())
        self.ax.set_ylim(self.final_tree.getYMin(), self.final_tree.getYMax() + 1)

        self.width = self.final_tree.width
        self.height = self.final_tree.height

        self.drawNode(self.final_root, renderer)

        self.fig.show()
