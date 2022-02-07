# -*- coding: utf8 -*-

import numpy as np


def load_dataset():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5, [2, 5]]]


# create a collection c1, de-duplicate the dataset, sort it, and
# put it into list, then convert all elements into frozenset
def create_collection(dataset):
    coll = []
    for txn in dataset:
        for item in txn:
            # de-duplicate it
            if [item] not in coll:
                coll.append([item])
    # sort it
    coll.sort()
    # the elements of frozenset are stable, they can be as the key of dictionary
    return map(frozenset, coll)


# calculate the support of candidate set `candidates` in dataset `dataset`,
# and return the data whose support is larger than the minimum support
def scan_data(dataset, candidates, min_support):
    # ss_cnt temporarily stores the frequency of `candidates`
    ss_cnt = {}
    for tid in dataset:
        for cand in candidates:
            # check if all elements of `cand` is in the `tid`
            if cand.issubset(tid):
                if cand not in ss_cnt:
                    ss_cnt[cand] = 1
                else:
                    ss_cnt[cand] += 1
    # the length of dataset
    num_items = len(dataset)
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        # support = count(key) / num_items
        support = ss_cnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


# enter the frequency itemset `freq_itemset` and the number of items `num_item`
# to get the all possible candidate itemset `candidates`
def apriori_gen(freq_itemset, num_item):
    ret_list = []
    num_freq_itemset = len(freq_itemset)
    for i in range(num_freq_itemset):
        for j in range(i + 1, num_freq_itemset):
            l1 = list(freq_itemset[i])[: num_item - 2]
            l2 = list(freq_itemset[j])[: num_item - 2]
            l1.sort()
            l2.sort()
            # if first k-2 elements are equal
            if l1 == l2:
                # set union
                ret_list.append(freq_itemset[i] | freq_itemset[j])
    return ret_list


# find the frequency item set
def apriori(dataset, min_support=0.5):
    """
    firstly, create a collection `c1`
    than, scan the dataset to determine whether these item sets with only one element
          meet the requirements of the minimum support.
          those item sets meeting the minimum support requirements constitutes the set `l1`
    thirdly, the elements in the `l1` combined into `c2`, and `c2` is further filtered into `l2`, and so on.
             Until the length of `candidates` become 0, the support of all frequent item sets can be found.
    """
    c1 = create_collection(dataset)
    data = list(map(set, dataset))
    l1, support_data = scan_data(data, c1, min_support)

    l, num_item = [l1], 2
    while len(l[num_item - 2]) > 0:
        candidates = apriori_gen(l[num_item - 2], num_item)

        itemset, support = scan_data(data, candidates, min_support)
        support_data.update(support)
        if len(itemset) == 0:
            break
        l.append(itemset)
        num_item += 1
    return l, support_data

