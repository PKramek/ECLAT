from collections import namedtuple
from itertools import combinations
from time import time
from typing import Tuple, List

import numpy as np
import pandas as pd

Candidate = namedtuple('Candidate', ['products', 'tidlist'])


def read_and_convert_data(file_name: str, separator=' '):
    df = pd.read_csv(file_name, names=['transactions'])
    # Converting each row to numpy int array
    df['transactions'] = df['transactions'].apply(lambda x: np.fromstring(x, dtype=np.int, sep=separator))

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


def create_frequent_itemsets_index(frequent_itemset: List[Tuple[np.ndarray, int]]):
    index = {}
    for row in frequent_itemset:
        for itemset_information in row:
            index[np.array_str(itemset_information[0])] = itemset_information[1]

    return index


def not_empty_subsets_generator(a: np.ndarray):
    length = len(a)
    for i in range(1, length):
        yield from combinations(a, r=i)


def association_rule_generator(itemset: np.ndarray):
    a_as_set = set(itemset)
    for antecedent in not_empty_subsets_generator(itemset):
        antecedent = set(antecedent)
        consequent = a_as_set - antecedent

        antecedent = np.array(list(antecedent), dtype=int)
        consequent = np.array(list(consequent), dtype=int)
        if antecedent.shape != ():
            assert all(np.equal(antecedent, sorted(antecedent)))
        if consequent.shape != ():
            assert all(np.equal(consequent, sorted(consequent)))

        yield antecedent, consequent


def get_itemset_support(frequet_itemsets, itemset):
    return frequet_itemsets[np.array_str(itemset)]


time_start = time()
dataset = read_and_convert_data('data/BMS1_itemset_mining.txt')
exec_time = time() - time_start
print("exec time: {}".format(exec_time))

min_support = 100

frequet_itemsets = eclat(dataset, min_support)
itemsets_index = create_frequent_itemsets_index(frequet_itemsets)
print(frequet_itemsets[3][0][0])
print(type(frequet_itemsets[3][0][0]))

rule_generator = association_rule_generator(frequet_itemsets[3][0][0])
first_ant, first_con = next(rule_generator)

print("{} -> {}, supports: {}, {}".format(first_ant, first_con, get_itemset_support(itemsets_index, first_ant),
                                          get_itemset_support(itemsets_index, first_con)))
# TODO change way index are created for length 1 or change way ante or cons are created for length 1
