# -*- coding: utf8 -*-
import numpy as np
from collections import defaultdict


class TreeNode:
    def __init__(self, name_val, num_occur, parent_node):
        self.name = name_val
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node  # need to be updated
        self.children = {}

    def inc(self, num_occur):
        self.count += num_occur

    def disp(self, ind=1):
        print('\t' * ind, '{}:{}'.format(self.name, self.count))
        for child in self.children.values():
            child.disp(ind + 1)


def create_tree(dataset, min_support=1):
    # {elem: occur_count}, which elements are larger than minimum support
    header_table = {}
    for txn in dataset:
        for item in txn:
            header_table[item] = header_table.get(item, 0) + dataset[txn]
    # remove the elements whose support isn't larger than minimum support
    for k in list(header_table.keys()):
        if header_table[k] < min_support:
            del header_table[k]
    freq_item_set = set(header_table.keys())

    # if no elements meet up the requirement, return directly
    if len(freq_item_set) == 0:
        return None, None
    for k in header_table:
        # format dict{ key: [count, None]}
        header_table[k] = [header_table[k], None]

    # create tree
    ret_tree = TreeNode('Null Set', 1, None)
    for txn, count in dataset.items():
        local = {}
        for item in txn:
            if item in freq_item_set:
                local[item] = header_table[item][0]
        if len(local) > 0:
            ordered_items = [v[0] for v in sorted(local.items(), key=lambda p: p[1], reverse=True)]
            update_tree(ordered_items, ret_tree, header_table, count)
    return ret_tree, header_table


def update_tree(items, tree, header_table, count):
    if items[0] in tree.children:
        tree.children[items[0]].inc(count)
    else:
        tree.children[items[0]] = TreeNode(items[0], count, tree)
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1:], tree.children[items[0]], header_table, count)


def update_header(node_to_test, target_node):
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node


def load_simple_data():
    simp_data = [['r', 'z', 'h', 'j', 'p'],
                 ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                 ['z'],
                 ['r', 'x', 'n', 'o', 's'],
                 ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                 ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simp_data


def create_init_set(dataset):
    ret_dict = {}
    for txn in dataset:
        if frozenset(txn) not in ret_dict.keys():
            ret_dict[frozenset(txn)] = 1
        else:
            ret_dict[frozenset(txn)] += 1
    return ret_dict
