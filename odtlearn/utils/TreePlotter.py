import numpy as np
from seaborn import color_palette
from sklearn.tree._export import _MPLTreeExporter

from odtlearn.utils._reingold_tilford import Tree, buchheim


class MPLPlotter(_MPLTreeExporter):
    """
    A class for plotting optimal decision trees using Matplotlib.

    Parameters
    ----------
    tree : object
        The decision tree object to be plotted.
    node_dict : dict
        A dictionary containing information about each node in the tree.
    column_names : list
        A list of feature names used in the decision tree.
    max_depth : int
        The maximum depth of the tree to plot.
    classes : list
        A list of class names for the target variable.
    model_name : str
        The name of the model used to generate the decision tree.
    label : str, optional (default="all")
        The type of labels to display on the nodes. Can be "all", "root", or None.
    filled : bool, optional (default=False)
        Whether to fill the nodes with color.
    rounded : bool, optional (default=False)
        Whether to use rounded corners for the node boxes.
    precision : int, optional (default=3)
        The number of decimal places to display for the cutoff values.
    fontsize : int or None, optional (default=None)
        The font size for the node labels. If None, the fontsize is automatically adjusted.
    color_dict : dict, optional
        A dictionary specifying the colors for nodes and leaves.
        Default: {"node": None, "leaves": []}
    edge_annotation : bool, optional (default=True)
        Whether to display annotations on the edges.
    arrow_annotation_font_scale : float, optional (default=0.5)
        The font scale for the arrow annotations.
    debug : bool, optional (default=False)
        Whether to print debug information.

    Attributes
    ----------
    classes : list
        A list of class names for the target variable.
    max_depth : int
        The maximum depth of the tree to plot.
    tree : object
        The decision tree object to be plotted.
    node_dict : dict
        A dictionary containing information about each node in the tree.
    column_names : list
        A list of feature names used in the decision tree.
    class_names : list
        A list of class names for the target variable.
    color_options : list
        A list of color options for the nodes and leaves.
    color_dict : dict
        A dictionary specifying the colors for nodes and leaves.
    edge_annotation : bool
        Whether to display annotations on the edges.
    arrow_annotation_font_scale : float
        The font scale for the arrow annotations.
    debug : bool
        Whether to print debug information.
    model_name : str
        The name of the model used to generate the decision tree.
    characters : list
        A list of special characters used for formatting the node labels.
    bbox_args : dict
        A dictionary of arguments for the bounding box of the nodes.
    arrow_args : dict
        A dictionary of arguments for the arrow style.

    Methods
    -------
    get_fill_color(node_id)
        Returns the fill color for a given node.
    node_to_str(node_id, leaf, selected_feature, cutoff, value)
        Converts a node to a string representation.
    export(ax=None, distance=1.0)
        Exports the decision tree plot to a Matplotlib axis.
    recurse(node, ax, max_x, max_y, depth=0)
        Recursively plots the nodes and edges of the decision tree.
    """

    def __init__(
        self,
        tree,
        node_dict,
        column_names,
        max_depth,
        classes,
        model_name,
        label="all",
        filled=False,
        rounded=False,
        precision=3,
        fontsize=None,
        color_dict={
            "node": None,
            "leaves": [],
        },
        edge_annotation=True,
        arrow_annotation_font_scale=0.5,
        debug=False,
    ):
        self.classes = classes
        self.max_depth = max_depth
        self.tree = tree
        self.node_dict = node_dict
        self.column_names = column_names
        self.class_names = classes
        self.color_options = [
            [int(value * 255) for value in color]
            for color in color_palette("husl", len(self.classes) + 1)
        ]
        # self.color_options = _color_brew(len(self.classes) + 1)
        self.color_dict = color_dict
        if self.color_dict["node"] is None:
            self.color_dict["node"] = self.color_options[-1]
        if len(self.color_dict["leaves"]) == 0:
            self.color_dict["leaves"] = self.color_options[:-1]
        self.edge_annotation = edge_annotation
        self.arrow_annotation_font_scale = arrow_annotation_font_scale
        self.debug = debug
        self.model_name = model_name

        super().__init__(
            max_depth=self.max_depth,
            feature_names=self.column_names,
            class_names=self.class_names,
            label=label,
            filled=filled,
            impurity=False,
            node_ids=False,
            proportion=False,
            rounded=rounded,
            precision=precision,
            fontsize=fontsize,
        )

        # The colors to render each node with
        # self.colors = {"bounds": None}

        self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"

        self.arrow_args = dict(arrowstyle="<-")

    def get_fill_color(self, node_id):
        # Fetch appropriate color for node
        # get value for particular node and see if that works?
        # pruned, branching, selected_feature, cutoff, leaf, value
        _, branching, _, _, leaf, value = self.node_dict[node_id]
        alpha = 1
        if leaf:
            if self.debug:
                print(f"Node id: {node_id}")
                print(f"Leaf value: {value}")
                print(f"value passed to color_dict: {int(value - 1)}")
                print(self.color_dict)
            color = self.color_dict["leaves"][int(value - 1)]
        if branching:
            color = self.color_dict["node"]

        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)

    def node_to_str(self, node_id, leaf, selected_feature, cutoff, value):
        characters = self.characters
        node_string = characters[-1]

        name = value if selected_feature is None else selected_feature
        if type(name) not in [str, np.str_]:
            if type(name) != int:
                name = str(int(name))
            else:
                name = str(name)
        cutoff = cutoff if selected_feature is not None else None

        # Should labels be shown?
        labels = (self.label == "root" and node_id == 0) or self.label == "all"
        if self.node_ids:
            if labels:
                node_string += "node "
            node_string += characters[0] + str(node_id) + characters[4]
        if not leaf:
            # then we want to write the selected feature and if applicable the cutoff
            feature = name
            if cutoff is not None:
                # = or <=
                sign = (
                    "="
                    # if self.model_name in ["FairOCT", "FlowOCT", "BendersOCT"]
                    # else characters[3]
                )
                node_string += (
                    f"{feature} {sign} {round(cutoff, self.precision)}{characters[4]}"
                )
                if self.debug:
                    print(f"cutoff value: {cutoff}")
                    print(f"rounded cutoff value: {round(cutoff, self.precision)}")
            else:
                node_string += f"{feature}"
        else:
            if labels:
                if "OPT" in self.model_name:
                    node_string += "trt = "
                else:
                    node_string += "class = "
            class_name = f"{name}"
            node_string += class_name
        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]

        return node_string + characters[5]

    # now we need to get our tree into a form sklearn can work with
    def _make_tree(self, node_id, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        _, _, selected_feature, cutoff, leaf, value = self.node_dict[node_id]
        label = self.node_to_str(node_id, leaf, selected_feature, cutoff, value)
        if not leaf and depth <= self.max_depth:
            left_child = self.tree.get_left_children(node_id)
            right_child = self.tree.get_right_children(node_id)
            children = [
                self._make_tree(left_child, depth=depth + 1),
                self._make_tree(right_child, depth=depth + 1),
            ]
        else:
            return Tree(label, node_id)
        return Tree(label, node_id, *children)

    def export(self, ax=None, distance=1.0):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(1)
        draw_tree = buchheim(my_tree, distance=distance)
        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        self.recurse(draw_tree, ax, max_x, max_y)

        anns = [
            ann
            for ann in ax.get_children()
            if isinstance(ann, Annotation) and ann.get_text() not in ["yes", "no"]
        ]
        arrow_texts = [
            ann
            for ann in ax.get_children()
            if isinstance(ann, Annotation) and ann.get_text() in ["yes", "no"]
        ]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [ann.get_bbox_patch().get_window_extent() for ann in anns]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            size = anns[0].get_fontsize() * min(
                scale_x / max_width, scale_y / max_height
            )
            for ann in anns:
                ann.set_fontsize(size)

            for arrow in arrow_texts:
                arrow.set_fontsize(size * self.arrow_annotation_font_scale)

        # return anns

    def recurse(self, node, ax, max_x, max_y, depth=0):
        import matplotlib.pyplot as plt

        kwargs = dict(
            bbox=self.bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=self.arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        arrow_kwargs = dict(
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=1,
                mutation_scale=0.5,
                pad=0.5,
            ),
            ha="center",
            va="center",
            zorder=100,
            xycoords="axes fraction",
        )

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize
            arrow_kwargs["fontsize"] = self.fontsize * self.arrow_annotation_font_scale

        # offset things by .5 to center them in plot
        offset_amt = 0.5
        xy = ((node.x + offset_amt) / max_x, (max_y - node.y - offset_amt) / max_y)

        # when we annotate a node, we can add a text annotation at the midpoint between parent and node
        # it needs to be offset to the left or right depending on whether the node is a left child or right child
        # each node has a _lmost_sibling property that we can use to check if it is the left or right node
        # if _lmost_sibling is None it is the left node

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                # need to adapt get_fill_color
                kwargs["bbox"]["fc"] = self.get_fill_color(node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + offset_amt) / max_x,
                    (max_y - node.parent.y - offset_amt) / max_y,
                )
                if self.edge_annotation:
                    # calculate midpoint between parent and current node
                    midpoint = [(xy_parent[0] + xy[0]) / 2, (xy_parent[1] + xy[1]) / 2]
                    # check if left-most sibling
                    if node._lmost_sibling is None:
                        midpoint[0] = midpoint[0]
                        arrowtext = "yes"
                    else:
                        midpoint[0] = midpoint[0]
                        arrowtext = "no"
                    # use text instead of annotation because we use annotations later for determining max extent of plot
                    # ax.text(midpoint[0], midpoint[1], arrowtext)
                    ax.annotate(arrowtext, xy=midpoint, **arrow_kwargs)

                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, ax, max_x, max_y, depth=depth + 1)

        else:
            xy_parent = (
                (node.parent.x + offset_amt) / max_x,
                (max_y - node.parent.y - offset_amt) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)
