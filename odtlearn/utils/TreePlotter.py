from sklearn.tree._export import _MPLTreeExporter, _color_brew
from sklearn.tree._reingold_tilford import Tree, buchheim
from odtlearn.utils.StrongTreeUtils import get_node_status


class MPLPlotter(_MPLTreeExporter):
    def __init__(
        self,
        grb_model,
        column_names,
        b,
        w,
        p,
        max_depth,
        classes,
        label="all",
        filled=False,
        rounded=False,
        precision=3,
        fontsize=None,
        color_dict={"node": None, "leaves": []},
    ):
        self.classes = classes
        self.max_depth = max_depth
        self.tree = grb_model.tree
        self.grb_model = grb_model
        self.column_names = column_names
        self.feature_names = column_names
        self.class_names = classes
        self.b = b
        self.w = w
        self.p = p
        self.color_options = _color_brew(len(self.classes) + 1)
        self.color_dict = color_dict
        if self.color_dict["node"] is None:
            self.color_dict["node"] = self.color_options[-1]
        if len(self.color_dict["leaves"]) == 0:
            self.color_dict["leaves"] = self.color_options[:-1]

        super().__init__(
            max_depth=self.max_depth,
            feature_names=self.feature_names,
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
        _, branching, _, leaf, value = get_node_status(
            self.grb_model, self.b, self.w, self.p, node_id
        )
        alpha = 1
        if leaf:
            print(int(value - 1))
            print(self.color_dict["leaves"])
            color = self.color_dict["leaves"][int(value - 1)]
        if branching:
            color = self.color_dict["node"]

        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)

    # now we need to get our tree into a form sklearn can work with
    def _make_tree(self, node_id, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        _, _, selected_feature, leaf, value = get_node_status(
            self.grb_model, self.b, self.w, self.p, node_id
        )
        name = str(value) if selected_feature is None else selected_feature
        if not leaf and depth <= self.max_depth:
            left_child = self.tree.get_left_children(node_id)
            right_child = self.tree.get_right_children(node_id)
            children = [
                self._make_tree(left_child, depth=depth + 1),
                self._make_tree(right_child, depth=depth + 1),
            ]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

    def export(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(1)
        draw_tree = buchheim(my_tree)
        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        # need to modify recurse for our purposes!!!
        self.recurse(draw_tree, ax, max_x, max_y)

        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

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

        return anns

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

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        # offset things by .5 to center them in plot
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

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
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, ax, max_x, max_y, depth=depth + 1)

        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)
