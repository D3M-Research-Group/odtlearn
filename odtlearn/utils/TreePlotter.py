from sklearn.tree._export import _MPLTreeExporter, _color_brew
from odtlearn.utils._reingold_tilford import Tree, buchheim
import numpy as np


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
        color_dict={
            "node": None,
            "leaves": [],
        },  # TO-DO: document behavior of this dict
        edge_annotation=True,
        arrow_annotation_font_scale=0.5,
        debug=False,
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
        self.edge_annotation = edge_annotation
        self.arrow_annotation_font_scale = arrow_annotation_font_scale
        self.debug = debug

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
        # pruned, branching, selected_feature, cutoff, leaf, value
        _, branching, _, _, leaf, value = self.get_node_status(
            self.grb_model, self.b, self.w, self.p, node_id
        )
        alpha = 1
        if leaf:
            if self.debug:
                print(f"Leaf value: {value}")
                print(f"value passed to color_dict: {int(value - 1)}")
            color = self.color_dict["leaves"][int(value - 1)]
        if branching:
            color = self.color_dict["node"]

        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)

    def get_node_status(self, grb_model, b, w, p, n):
        """
        This function give the status of a given node in a tree. By status we mean whether the node
            1- is pruned? i.e., we have made a prediction at one of its ancestors
            2- is a branching node? If yes, what feature do we branch on
            3- is a leaf? If yes, what is the prediction at this node?

        Parameters
        ----------
        grb_model :
            The gurobi model solved to optimality (or reached to the time limit).
        b :
            The values of branching decision variable b.
        w :
            The values of prediction decision variable w.
        p :
            The values of decision variable p
        n :
            A valid node index in the tree

        Returns
        -------
        pruned : int
            pruned=1 iff the node is pruned
        branching : int
            branching = 1 iff the node branches at some feature f
        selected_feature : str
            The feature that the node branch on
        leaf : int
            leaf = 1 iff node n is a leaf in the tree
        value :  double
            if node n is a leaf, value represent the prediction at this node
        """

        pruned = False
        branching = False
        leaf = False
        value = None
        selected_feature = None
        cutoff = None

        model_type = grb_model.model.ModelName
        assert len(model_type) > 0

        # Node status for StrongTree and FairTree
        if model_type in ["FairOCT", "FlowOCT", "BendersOCT"]:
            cutoff = 0
            p_sum = 0
            for m in grb_model.tree.get_ancestors(n):
                p_sum = p_sum + p[m]
            if p[n] > 0.5:  # leaf
                leaf = True
                for k in grb_model.labels:
                    if w[n, k] > 0.5:
                        value = k
            elif p_sum == 1:  # Pruned
                pruned = True

            if n in grb_model.tree.Nodes:
                if (pruned is False) and (leaf is False):  # branching
                    for f in grb_model.X_col_labels:
                        if b[n, f] > 0.5:
                            selected_feature = f
                            branching = True
        elif model_type == "RobustOCT":
            p_sum = 0
            for m in grb_model.tree.get_ancestors(n):
                # need to sum over all w values for a given n
                p_sum += sum(w[m, k] for k in grb_model.labels)
            # to determine if a leaf, we look at its w value
            # and find for which label the value > 0.5
            col_idx = np.asarray([w[n, k] > 0.5 for k in grb_model.labels]).nonzero()[0]
            # col_idx = np.asarray(w[n, :] > 0.5).nonzero().flatten()
            # assuming here that we can only have one column > 0.5
            if len(col_idx) > 0:
                leaf = True
                value = grb_model.labels[int(col_idx[0])]
            elif p_sum == 1:
                pruned = True

            if not pruned and not leaf:
                for f, theta in grb_model.f_theta_indices:
                    if b[n, f, theta] > 0.5:
                        selected_feature = f
                        cutoff = theta
                        branching = True

        return pruned, branching, selected_feature, cutoff, leaf, value

    def node_to_str(self, node_id, leaf, selected_feature, cutoff, value):
        characters = self.characters
        node_string = characters[-1]

        name = str(value) if selected_feature is None else selected_feature
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
                node_string += "feature %s %s %s%s" % (
                    feature,
                    "="
                    if self.grb_model.model.ModelName
                    in ["FairOCT", "FlowOCT", "BendersOCT"]
                    else characters[3],
                    round(cutoff, self.precision),
                    characters[4],
                )
            else:
                node_string += "%s" % (feature)
        else:
            if labels:
                node_string += "class = "
            class_name = "%s" % (name)
            node_string += class_name
        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]

        return node_string + characters[5]

    # now we need to get our tree into a form sklearn can work with
    def _make_tree(self, node_id, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        _, _, selected_feature, cutoff, leaf, value = self.get_node_status(
            self.grb_model, self.b, self.w, self.p, node_id
        )
        # print(leaf, selected_feature, cutoff, value)
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

    def export(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(1)
        draw_tree = buchheim(my_tree, distance=1.0)
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
                # edgecolor="black",
                edgecolor="none",
                alpha=1,
                mutation_scale=0.5,
                pad=0.5
                # fill=False
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
