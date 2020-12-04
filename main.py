from collections import namedtuple
from copy import copy
from pprint import pprint
from typing import Tuple, List

import numpy as np
import pandas as pd

Candidate = namedtuple('Candidate', ['products', 'tidlist'])


def read_and_convert_data(file_name: str):
    df = pd.read_csv(file_name, names=['transactions'])
    # Converting each row to numpy int array
    df['transactions'] = df['transactions'].apply(lambda x: np.fromiter(map(int, x.split(' ')), dtype=np.int))

    return df


def get_tidlists(data_set) -> dict:
    tidlists = dict()

    for index, row in dataset.iterrows():
        row_type = type(row)
        for product in row['transactions']:
            tidlists.setdefault(product, []).append(index)

    return tidlists


def eclat(dataset, min_support: int):
    frequent_itemsets = []

    tidlists = get_tidlists(dataset)
    prev_frequent, prev_frequent_tidlists = get_frequent_L1(tidlists, min_support)
    add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)

    prev_frequent, prev_frequent_tidlists = get_frequent_L2(prev_frequent, prev_frequent_tidlists, min_support)
    add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)

    while len(prev_frequent) >= 2:
        prev_frequent, prev_frequent_tidlists = get_frequent_Lk(prev_frequent, prev_frequent_tidlists, min_support)
        add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)

    return frequent_itemsets


def get_frequent_L1(tidlists: dict, min_support: int) -> Tuple[List[int], List[np.ndarray]]:
    frequent_l1 = []
    frequent_l1_tidlists = []

    sorted_keys = sorted(tidlists.keys())
    for key in sorted_keys:
        if len(tidlists[key]) > min_support:
            frequent_l1.append(key)
            frequent_l1_tidlists.append(np.array(tidlists[key]))

    return frequent_l1, frequent_l1_tidlists


def get_frequent_L2(frequent_l1: List[int], frequent_l1_tidlists: List[np.ndarray], min_support: int):
    frequent_l2 = []
    frequent_l2_tidlists = []

    for i in range(len(frequent_l1) - 1):
        for j in range(i + 1, len(frequent_l1)):
            tidlist = np.intersect1d(frequent_l1_tidlists[i], frequent_l1_tidlists[j])
            if len(tidlist) > min_support:
                frequent_l2.append([frequent_l1[i], frequent_l1[j]])
                frequent_l2_tidlists.append(tidlist)

    return frequent_l2, frequent_l2_tidlists


def get_frequent_Lk(frequent_lk_1: List[List[int]], frequent_lk_1_tidlists: List[np.array], min_support: int):
    frequent_lk = []
    frequent_lk_tidlists = []
    itemset_len = len(frequent_lk_1[0])

    for i in range(len(frequent_lk_1) - 1):
        for j in range(i + 1, len(frequent_lk_1_tidlists)):

            first_except_last_item = np.delete(frequent_lk_1[i], itemset_len - 1)
            second_except_last_item = np.delete(frequent_lk_1[j], itemset_len - 1)
            first_last = frequent_lk_1[i][itemset_len - 1]
            second_last = frequent_lk_1[j][itemset_len - 1]

            if first_last != second_last and all(np.equal(first_except_last_item, second_except_last_item)):
                tidlist = np.intersect1d(frequent_lk_1_tidlists[i], frequent_lk_1_tidlists[j])
                if len(tidlist) > min_support:
                    new_candidate_itemset = copy(frequent_lk_1[i])
                    new_candidate_itemset.append(second_last)

                    frequent_lk.append(new_candidate_itemset)
                    frequent_lk_tidlists.append(tidlist)
            else:
                break

    return frequent_lk, frequent_lk_tidlists


def generate_frequent_candidates(prev_frequent_candidates: dict, min_support: int):
    sorted_keys = sorted(prev_frequent_candidates.keys())
    candidates = dict()
    for i in range(len(sorted_keys) - 1):
        for j in range(i, len(sorted_keys) - 1):
            assert i != j
            candidate = sorted_keys[i] + " " + sorted_keys[i + 1]
            first_tidlist = prev_frequent_candidates[sorted_keys[i]]
            second_tidlist = prev_frequent_candidates[sorted_keys[i + 1]]
            candidate_tidlist = first_tidlist & second_tidlist
            if len(candidate_tidlist) > min_support:
                candidates[candidate] = candidate_tidlist
    return candidates


def add_results(frequent_itemsets, frequent_candidates, frequent_candidates_tidlists):
    results = [(x, len(y)) for x, y in zip(frequent_candidates, frequent_candidates_tidlists)]
    frequent_itemsets.append(results)

    return frequent_itemsets


def get_frequent_candidates(candidates: dict, min_support):
    frequent_candidates = {}
    dict_keys = candidates.keys()
    for key in dict_keys:
        if len(candidates[key]) <= min_support:
            frequent_candidates[key] = candidates[key]

    return frequent_candidates


dataset = read_and_convert_data('data/BMS1_itemset_mining.txt')
tidlists = get_tidlists(dataset)
min_support = 100

frequet_itemsets = eclat(dataset, min_support)
pprint(frequet_itemsets)
