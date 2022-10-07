# flake8: noqa
import re
import sys
from itertools import zip_longest

import pydot

if len(sys.argv) > 1:
    input_filename, output_filename = sys.argv[1:]
else:
    input_filename = "classes.dot"
    output_filename = "classes.png"

graph = pydot.graph_from_dot_file(input_filename)[0]
graph.del_node('"\\n"')
EXCLUDED_SUFFIX_ATTR_RE = re.compile(
    r"(_dtypes|_col_labels|_indices|values|features|name|_P)$"
)
EXCLUDED_PREFIX_ATTR_RE = re.compile(
    r"^(get_|fairness_metric_summary|_add_fairness_constraint)"
)


def make_row(value1, value2, border=False):
    if border:
        return f'<tr><td align="left" border="1" sides="B">{value1} <br align="left" /></td> <td align="right" border="1" sides="B">{value2} </td></tr>'
    else:
        return f'<tr><td align="left">{value1} <br align="left" /></td> <td align="right">{value2} </td></tr>'


def make_end_row(value, border=True):
    if border:
        return f'<tr > <td align="left" colspan="2" border="1" sides="B"> {value} </td></tr>'
    else:
        return f'<tr > <td align="left" colspan="2"> {value} </td></tr>'


def make_header(name):
    return f'<tr> <td align="center" colspan="2" border="2" sides="B"><b>{name}</b> </td></tr>'


def make_table(name, attributes, methods):
    table = ["<", '<table border="0">']
    table.append(make_header(name))
    attribute_pairs = list(zip_longest(attributes[0::2], attributes[1::2]))
    attr_len = len(attribute_pairs)
    method_pairs = list(zip_longest(methods[0::2], methods[1::2]))
    method_len = len(method_pairs)
    for idx in range(attr_len):
        if idx == attr_len - 1:
            if attribute_pairs[idx][1] is None:
                table.append(make_end_row(attribute_pairs[idx][0], border=True))
            else:
                table.append(
                    make_row(
                        attribute_pairs[idx][0], attribute_pairs[idx][1], border=True
                    )
                )
        else:
            table.append(make_row(attribute_pairs[idx][0], attribute_pairs[idx][1]))
    for idx in range(method_len):
        if idx == method_len - 1:
            if method_pairs[idx][1] is None:
                table.append(make_end_row(method_pairs[idx][0], border=False))
            else:
                table.append(
                    make_row(method_pairs[idx][0], method_pairs[idx][1], border=False)
                )
        else:
            table.append(make_row(method_pairs[idx][0], method_pairs[idx][1]))
    table.append("</table>")
    table.append(">")
    return table


for node in graph.get_node_list():
    label = node.get_label()
    if label:
        if "Abstract Base Class" not in label:
            name, attributes, methods = label.split("|")
            name = name.replace('"{', "")
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
                attr_names = priv_attr_names + pub_attr_names
                attr_names = [
                    attr
                    for attr in attr_names
                    if not EXCLUDED_SUFFIX_ATTR_RE.search(attr)
                ]
                attributes = attr_names
                method_names = methods.split("\\l")[:-1]
                method_names = [
                    method
                    for method in method_names
                    if not EXCLUDED_PREFIX_ATTR_RE.search(method)
                ]
                methods = method_names
                label = "\n".join(make_table(name, attributes, methods))
                node.set_label(label)
            else:
                attributes = attributes.split("\\l")[:-1]
                methods = methods.split("\\l")[:-1]
                tmp_label = ["<", '<table border="0">']
                tmp_label.append(make_header(name))
                tmp_label.append(make_end_row(attributes[0]))
                tmp_label.append(make_end_row(methods[0], border=False))
                tmp_label.append("</table>")
                tmp_label.append(">")
                node.set_label("\n".join(tmp_label))
        else:
            name = label.replace('"{', "").replace('}"', "")
            node.set_label(f"< <b>{name}</b> >")


# for node in graph.get_node_list():
#     label = node.get_label()
#     if label:
#         if "Abstract Base Class" not in label:
#             name, attributes, methods = label.split("|")
#             if len(attributes) > 2:
#                 attr_names = attributes.split("\\l")[:-1]
#                 priv_attr_names = [
#                     attr.replace(" : ndarray, DataFrame", "").replace(
#                         " : DataFrame, ndarray", ""
#                     )
#                     for attr in attr_names
#                     if attr[0] == "_"
#                 ]
#                 pub_attr_names = [
#                     attr.replace(" : ndarray, DataFrame", "").replace(
#                         " : DataFrame, ndarray", ""
#                     )
#                     for attr in attr_names
#                     if attr[0] != "_"
#                 ]
#                 pub_attr_names.sort(key=len, reverse=True)
#                 priv_attr_names.sort(key=len, reverse=True)
#                 attr_names = priv_attr_names + pub_attr_names + [""]
#                 attr_names = [
#                     attr
#                     for attr in attr_names
#                     if not EXCLUDED_SUFFIX_ATTR_RE.search(attr)
#                 ]
#                 attributes = "\\l".join(attr_names)
#                 method_names = methods.split("\\l")[:-1]
#                 method_names = [
#                     method
#                     for method in method_names
#                     if not EXCLUDED_PREFIX_ATTR_RE.search(method)
#                 ]
#                 methods = "\\l".join(method_names + ['}"'])
#                 label = "|".join([name, attributes, methods])
#                 node.set_label(label)

graph.write_png(output_filename)
