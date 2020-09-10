# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# JDS: copied this from jax, made it self-contained
# Currently just used for improved_checkpoint
# pylint: disable=bad-indentation

import functools
import itertools as it
import collections


def unzip2(xys):
    xs = []
    ys = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)


def partial(fun, *args, **kwargs):
    wrapped = functools.partial(fun, *args, **kwargs)
    functools.update_wrapper(wrapped, fun)
    wrapped._bound_args = args  # pylint: disable=protected-access
    return wrapped

def concatenate(xs):
    return list(it.chain.from_iterable(xs))


def tree_map(f, tree):
    """Map a function over a pytree to produce a new pytree.

  Args:
    f: function to be applied at each leaf.
    tree: a pytree to be mapped over.

  Returns:
    A new pytree with the same structure as `tree` but with the value at each
    leaf given by `f(x)` where `x` is the value at the corresponding leaf in
    `tree`.
  """
    node_type = node_types.get(type(tree))
    if node_type:
        children, node_spec = node_type.to_iterable(tree)
        new_children = [tree_map(f, child) for child in children]
        return node_type.from_iterable(node_spec, new_children)
    else:
        return f(tree)


def tree_multimap(f, tree, *rest):
    """Map a multi-input function over pytree args to produce a new pytree.

  Args:
    f: function that takes `1 + len(rest)` arguments, to be applied at the
      corresponding leaves of the pytrees.
    tree: a pytree to be mapped over, with each leaf providing the first
      positional argument to `f`.
    *rest: a tuple of pytrees, each with the same structure as `tree`.

  Returns:
    A new pytree with the same structure as `tree` but with the value at each
    leaf given by `f(x, *xs)` where `x` is the value at the corresponding leaf
    in `tree` and `xs` is the tuple of values at corresponding leaves in `rest`.
  """
    # equivalent to prefix_multimap(f, tree_structure(tree), tree, *rest)
    node_type = node_types.get(type(tree))
    if node_type:
        children, node_spec = node_type.to_iterable(tree)
        all_children = [children]
        for other_tree in rest:
            # other_node_type = node_types.get(type(other_tree))
            # if node_type != other_node_type:
            #   raise TypeError('Mismatch: {} != {}'.format(other_node_type, node_type))
            other_children, other_node_data = node_type.to_iterable(other_tree)
            if other_node_data != node_spec:
                raise TypeError("Mismatch: {} != {}".format(other_node_data, node_spec))
            all_children.append(other_children)

        new_children = [tree_multimap(f, *xs) for xs in zip(*all_children)]
        return node_type.from_iterable(node_spec, new_children)
    else:
        return f(tree, *rest)


def tree_reduce(f, tree):
    flat, _ = tree_flatten(tree)
    return functools.reduce(f, flat)

def tree_all(tree):
    flat, _ = tree_flatten(tree)
    return all(flat)

def walk_pytree(f_node, f_leaf, tree):
    node_type = node_types.get(type(tree))
    if node_type:
        children, node_spec = node_type.to_iterable(tree)
        proc_children, child_specs = unzip2(
            [walk_pytree(f_node, f_leaf, child) for child in children]
        )
        tree_def = PyTreeDef(node_type, node_spec, child_specs)
        return f_node(proc_children), tree_def
    else:
        return f_leaf(tree), PyLeaf()

tree_flatten = partial(walk_pytree, concatenate, lambda x: [x])

class PyTreeDef(object):
    def __init__(self, node_type, node_data, children):
        self.node_type = node_type
        self.node_data = node_data
        self.children = children

    def __repr__(self):
        if self.node_data is None:
            data_repr = ""
        else:
            data_repr = "[{}]".format(self.node_data)

        return "PyTree({}{}, [{}])".format(
            self.node_type.name, data_repr, ",".join(safe_map(repr, self.children))
        )

    def __hash__(self):
        return hash((self.node_type, self.node_data, tuple(self.children)))

    def __eq__(self, other):
        if isinstance(other, PyLeaf):
            return False
        else:
            return (
                self.node_type == other.node_type
                and self.node_data == other.node_data
                and self.children == other.children
            )

    def __ne__(self, other):
        return not self == other


class PyLeaf(object):
    def __repr__(self):
        return "*"

    def __eq__(self, other):
        return isinstance(other, PyLeaf)


class NodeType(object):
    def __init__(self, name, to_iterable, from_iterable):
        self.name = name
        self.to_iterable = to_iterable
        self.from_iterable = from_iterable


node_types = {}


def register_pytree_node(py_type, to_iterable, from_iterable):
    assert py_type not in node_types
    node_types[py_type] = NodeType(str(py_type), to_iterable, from_iterable)


def tuple_to_iterable(xs):
    return xs, None


def tuple_from_iterable(_keys, xs):
    return tuple(xs)


def list_to_iterable(xs):
    return tuple(xs), None


def list_from_iterable(_keys, xs):
    return list(xs)


def dict_to_iterable(xs):
    keys = tuple(sorted(xs.keys()))
    return tuple(map(xs.get, keys)), keys


def dict_from_iterable(keys, xs):
    return dict(zip(keys, xs))


def none_to_iterable(_xs):
    return (), None


def none_from_iterable(_keys, _xs):
    return None


register_pytree_node(tuple, tuple_to_iterable, tuple_from_iterable)
register_pytree_node(list, list_to_iterable, list_from_iterable)
register_pytree_node(dict, dict_to_iterable, dict_from_iterable)
register_pytree_node(collections.OrderedDict, dict_to_iterable, dict_from_iterable)
register_pytree_node(type(None), none_to_iterable, none_from_iterable)
