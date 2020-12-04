import numpy as np
import pandas as pd


def read_and_convert_data(file_name: str):
    df = pd.read_csv(file_name, names=['transactions'])
    # Converting each row to numpy int array
    df['transactions'] = df['transactions'].apply(lambda x: np.fromiter(map(int, x.split(' ')), dtype=np.int))

    return df


def get_tidlists(data_set) -> dict:
    tidlists = dict()

    for index, row in dataset.iterrows():
        for product in row[0].split(' '):
            tidlists.setdefault(product, set()).add(index)

    return tidlists


def eclat(dataset, min_support: int):
    frequent_sets = []

    tidlists = get_tidlists(dataset)
    prev_candidates = {}

    for key in tidlists.keys():
        if len(tidlists[key]) > min_support:
            prev_candidates[str(key)] = tidlists[key]

    tidlists = None
    add_results(frequent_sets, prev_candidates)
    i = 1
    while len(prev_candidates) >= 2:
        print(i)
        i += 1
        prev_candidates = generate_frequent_candidates(prev_candidates, min_support)
        add_results(frequent_sets, prev_candidates)

    return frequent_sets


def generate_frequent_candidates(prev_frequent_candidates: dict, min_support: int):
    sorted_keys = sorted(prev_frequent_candidates.keys())
    candidates = dict()
    for i in range(len(sorted_keys) - 1):
        for j in range(i, len(sorted_keys) - 1):
            candidate = sorted_keys[i] + " " + sorted_keys[i + 1]
            first_tidlist = prev_frequent_candidates[sorted_keys[i]]
            second_tidlist = prev_frequent_candidates[sorted_keys[i + 1]]
            candidate_tidlist = first_tidlist & second_tidlist
            if len(candidate_tidlist) > min_support:
                candidates[candidate] = candidate_tidlist
    return candidates


def add_results(frequent_sets, frequent_candidates):
    candidates_and_their_supports = {}
    frequent_sets.append({key: len(value) for (key, value) in frequent_candidates.items()})


def get_frequent_candidates(candidates: dict, min_support):
    frequent_candidates = {}
    dict_keys = candidates.keys()
    for key in dict_keys:
        if len(candidates[key]) <= min_support:
            frequent_candidates[key] = candidates[key]

    return frequent_candidates


dataset = read_and_convert_data('data/BMS1_itemset_mining.txt')

print(dataset.head())
# candidates = eclat(dataset, 3)
#
# keys = []
# for i in range(len(candidates)):
#     print(len(candidates[i]))
#
# assert len(keys) == len(set(keys))
