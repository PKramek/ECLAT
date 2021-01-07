import logging
from typing import Tuple, List

import numpy as np
import pandas as pd


class Eclat:
    def __init__(self, dataset_path: str, min_support: float, separator: str = ' '):
        assert isinstance(dataset_path, str)

        self.dataset = None
        self.frequent_itemsets = None
        self.num_of_transactions = None
        self.results_dataframe = None

        self.min_support = min_support

        self.read_and_convert_data(dataset_path, separator)

    @property
    def min_support(self):
        return self._min_support

    @min_support.setter
    def min_support(self, min_support: float):
        if not isinstance(min_support, float) or not (0 <= min_support <= 1):
            raise ValueError("Minimal support must be float in interval [0, 1]")
        self._min_support = min_support

    def get_frequent_itemsets(self):
        logging.info('min tidlist len: {}'.format(int(self.min_support * self.num_of_transactions)))
        frequent_itemsets = []

        tidlists = self._get_tidlists()
        prev_frequent, prev_frequent_tidlists = self._get_frequent_1_itemsets(tidlists)
        logging.debug("Added L1, len = {}".format(len(prev_frequent)))
        self._add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)
        del tidlists

        prev_frequent, prev_frequent_tidlists = self._get_frequent_2_itemsets(prev_frequent, prev_frequent_tidlists)

        self._add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)
        logging.debug("Added L2, number of itemsets: {}".format(len(prev_frequent)))
        i = 2
        while len(prev_frequent) >= 2:
            prev_frequent, prev_frequent_tidlists = self._get_frequent_Lk(prev_frequent, prev_frequent_tidlists)
            self._add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)
            i += 1
            logging.debug("Added L{}, number of itemsets = {}".format(i, len(prev_frequent)))

        self.frequent_itemsets = frequent_itemsets
        return frequent_itemsets

    def _get_frequent_1_itemsets(self, tidlists: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        frequent_1 = []
        frequent_1_tidlists = []

        sorted_keys = sorted(tidlists.keys())
        for key in sorted_keys:
            support = len(tidlists[key]) / self.num_of_transactions
            if support > self.min_support:
                # It must be a list for convenient retrieving support from index.
                frequent_1.append(np.array([key], dtype=int))
                frequent_1_tidlists.append(tidlists[key])

        return frequent_1, frequent_1_tidlists

    def _get_frequent_2_itemsets(self, frequent_1: List[np.ndarray], frequent_1_tidlists: List[np.ndarray]):
        frequent_2 = []
        frequent_2_tidlists = []

        for i in range(len(frequent_1) - 1):
            for j in range(i + 1, len(frequent_1)):
                tidlist = frequent_1_tidlists[i] & frequent_1_tidlists[j]
                support = len(tidlist) / self.num_of_transactions
                if support > self.min_support:
                    first = frequent_1[i][0]
                    second = frequent_1[j][0]
                    frequent_2.append(np.array([first, second]))
                    frequent_2_tidlists.append(tidlist)

        return frequent_2, frequent_2_tidlists

    def _get_frequent_Lk(self, frequent_k_1: List[np.ndarray], frequent_k_1_tidlists: List[np.array]):
        frequent_k = []
        frequent_k_tidlists = []
        itemset_len = len(frequent_k_1[0])

        for i in range(len(frequent_k_1) - 1):
            for j in range(i + 1, len(frequent_k_1_tidlists)):
                if self._only_last_different(frequent_k_1[i], frequent_k_1[j]):
                    tidlist = frequent_k_1_tidlists[i] & frequent_k_1_tidlists[j]
                    support = len(tidlist) / self.num_of_transactions
                    if support > self.min_support:
                        second_last = frequent_k_1[j][itemset_len - 1]
                        new_candidate_itemset = np.append(np.copy(frequent_k_1[i]), second_last)

                        frequent_k.append(new_candidate_itemset)
                        frequent_k_tidlists.append(tidlist)
                else:
                    break

        return frequent_k, frequent_k_tidlists

    def _only_last_different(self, first: np.ndarray, second: np.ndarray):
        num_of_equals = np.sum(np.equal(first, second))
        length = len(first)

        return num_of_equals == len(first) - 1 and first[length - 1] != second[length - 1]

    def _add_results(self, frequent_itemsets, frequent_candidates, frequent_candidates_tidlists):
        results = [(x, len(y) / self.num_of_transactions) for x, y in
                   zip(frequent_candidates, frequent_candidates_tidlists)]
        frequent_itemsets.append(results)

        return frequent_itemsets

    def _get_tidlists(self) -> dict:
        assert self.dataset is not None
        tidlists = dict()

        for index, row in self.dataset.iterrows():
            for product in row['transactions']:
                tidlists.setdefault(product, set()).add(index)

        return tidlists

    def read_and_convert_data(self, dataset_path: str, separator: str):
        df = pd.read_csv(dataset_path, names=['transactions'])
        # Converting each row to numpy int array
        df['transactions'] = df['transactions'].apply(lambda x: np.fromstring(x, dtype=np.int, sep=separator))
        self.dataset = df
        self.num_of_transactions = self.dataset.shape[0]
