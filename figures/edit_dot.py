import re
import sys

import pydot

if len(sys.argv) > 0:
    input_filename, output_filename = sys.argv[1:]
else:
    input_filename = "classes.dot"
    output_filename = "classes.png"

graph = pydot.graph_from_dot_file(input_filename)[0]
EXCLUDED_ATTR_RE = re.compile(
    r"(_dtypes|_col_labels|_indices|values|features|name|_p|_P)$"
)
graph.del_node('"\\n"')

for node in graph.get_node_list():
    label = node.get_label()
    if label:
        if "Abstract Base Class" not in label:
            print(label)
            name, attributes, methods = label.split("|")
            if len(attributes) > 2:
                attr_names = attributes.split("\\l")[:-1]
                priv_attr_names = [
                    attr.replace(" : ndarray, DataFrame", "").replace(
                        " : DataFrame, ndarray", ""
                    )
                    for attr in attr_names
                    if attr[0] == "_"
                ]
                pub_attr_names = [
                    attr.replace(" : ndarray, DataFrame", "").replace(
                        " : DataFrame, ndarray", ""
                    )
                    for attr in attr_names
                    if attr[0] != "_"
                ]
                pub_attr_names.sort(key=len, reverse=True)
                priv_attr_names.sort(key=len, reverse=True)
                attr_names = priv_attr_names + pub_attr_names + [""]
                attr_names = [
                    attr for attr in attr_names if not EXCLUDED_ATTR_RE.search(attr)
                ]
                attributes = "\\l".join(attr_names)
                label = "|".join([name, attributes, methods])
                node.set_label(label)

graph.write_png(output_filename)
