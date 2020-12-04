from collections import namedtuple
from time import time
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


def get_frequent_L1(tidlists: dict, min_support: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    frequent_l1 = []
    frequent_l1_tidlists = []

    sorted_keys = sorted(tidlists.keys())
    for key in sorted_keys:
        if len(tidlists[key]) > min_support:
            frequent_l1.append(np.array(key))
            frequent_l1_tidlists.append(np.array(tidlists[key]))

    return frequent_l1, frequent_l1_tidlists


def get_frequent_L2(frequent_l1: List[np.ndarray], frequent_l1_tidlists: List[np.ndarray], min_support: int):
    frequent_l2 = []
    frequent_l2_tidlists = []

    for i in range(len(frequent_l1) - 1):
        for j in range(i + 1, len(frequent_l1)):
            tidlist = np.intersect1d(frequent_l1_tidlists[i], frequent_l1_tidlists[j], assume_unique=True)
            if len(tidlist) > min_support:
                first = frequent_l1[i]
                second = frequent_l1[j]
                frequent_l2.append(np.array([first, second]))
                frequent_l2_tidlists.append(tidlist)

    return frequent_l2, frequent_l2_tidlists


def get_frequent_Lk(frequent_lk_1: List[np.ndarray], frequent_lk_1_tidlists: List[np.array], min_support: int):
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
                tidlist = np.intersect1d(frequent_lk_1_tidlists[i], frequent_lk_1_tidlists[j], assume_unique=True)
                if len(tidlist) > min_support:
                    new_candidate_itemset = np.append(np.copy(frequent_lk_1[i]), second_last)

                    frequent_lk.append(new_candidate_itemset)
                    frequent_lk_tidlists.append(tidlist)
            else:
                break

    return frequent_lk, frequent_lk_tidlists


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


time_start = time()

dataset = read_and_convert_data('data/BMS1_itemset_mining.txt')
tidlists = get_tidlists(dataset)
min_support = 50

frequet_itemsets = eclat(dataset, min_support)
exec_time = time() - time_start
print("exec time: {}".format(exec_time))
