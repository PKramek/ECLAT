from itertools import combinations
from math import sqrt
from typing import Tuple, List

import numpy as np
import pandas as pd


class Eclat:
    def __init__(self, dataset_path: str, min_support: float, min_confidence: float, separator: str = ' '):
        assert isinstance(dataset_path, str)

        self.dataset = None
        self.frequent_itemsets = None
        self.frequent_itemsets_index = None
        self.num_of_transactions = None

        self.min_support = min_support
        self.min_confidence = min_confidence

        self.read_and_convert_data(dataset_path, separator)

        self.results_dataframe = None

    @property
    def min_support(self):
        return self._min_support

    @min_support.setter
    def min_support(self, min_support: float):
        if not isinstance(min_support, float) or not (0 <= min_support <= 1):
            raise ValueError("Minimal support must be integer bigger than 1")
        self._min_support = min_support

    @property
    def min_confidence(self):
        return self._min_confidence

    @min_confidence.setter
    def min_confidence(self, min_confidence: float):
        if not isinstance(min_confidence, float) or not 0 < min_confidence < 1:
            raise ValueError("Minimal confidence must be floating point number in interval (0, 1)")

        self._min_confidence = min_confidence

    def eclat(self):
        frequent_itemsets = []

        tidlists = self._get_tidlists()
        prev_frequent, prev_frequent_tidlists = self._get_frequent_L1(tidlists)
        self._add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)

        prev_frequent, prev_frequent_tidlists = self._get_frequent_L2(prev_frequent, prev_frequent_tidlists)

        self._add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)

        while len(prev_frequent) >= 2:
            prev_frequent, prev_frequent_tidlists = self._get_frequent_Lk(prev_frequent, prev_frequent_tidlists)
            self._add_results(frequent_itemsets, prev_frequent, prev_frequent_tidlists)

        self.frequent_itemsets = frequent_itemsets
        self._create_frequent_itemsets_index()
        return frequent_itemsets

    def _get_frequent_L1(self, tidlists: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        frequent_l1 = []
        frequent_l1_tidlists = []

        sorted_keys = sorted(tidlists.keys())
        for key in sorted_keys:
            support = len(tidlists[key]) / self.num_of_transactions
            if support > self.min_support:
                # It must be a list for convenient retrieving support from index.
                frequent_l1.append(np.array([key], dtype=int))
                frequent_l1_tidlists.append(np.array(tidlists[key]))

        return frequent_l1, frequent_l1_tidlists

    def _get_frequent_L2(self, frequent_l1: List[np.ndarray], frequent_l1_tidlists: List[np.ndarray]):
        frequent_l2 = []
        frequent_l2_tidlists = []

        for i in range(len(frequent_l1) - 1):
            for j in range(i + 1, len(frequent_l1)):
                tidlist = np.intersect1d(frequent_l1_tidlists[i], frequent_l1_tidlists[j], assume_unique=True)
                support = len(tidlist) / self.num_of_transactions
                if support > self.min_support:
                    first = frequent_l1[i][0]
                    second = frequent_l1[j][0]
                    frequent_l2.append(np.array([first, second]))
                    frequent_l2_tidlists.append(tidlist)

        return frequent_l2, frequent_l2_tidlists

    def _get_frequent_Lk(self, frequent_lk_1: List[np.ndarray], frequent_lk_1_tidlists: List[np.array]):
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
                    support = len(tidlist) / self.num_of_transactions
                    if support > self.min_support:
                        new_candidate_itemset = np.append(np.copy(frequent_lk_1[i]), second_last)

                        frequent_lk.append(new_candidate_itemset)
                        frequent_lk_tidlists.append(tidlist)
                else:
                    break

        return frequent_lk, frequent_lk_tidlists

    def _add_results(self, frequent_itemsets, frequent_candidates, frequent_candidates_tidlists):
        results = [(x, len(y) / self.num_of_transactions) for x, y in
                   zip(frequent_candidates, frequent_candidates_tidlists)]
        frequent_itemsets.append(results)

        return frequent_itemsets

    def _create_frequent_itemsets_index(self) -> None:
        index = {}
        for row in self.frequent_itemsets:
            for itemset_information in row:
                index[np.array_str(itemset_information[0])] = itemset_information[1]

        self.frequent_itemsets_index = index

    def _get_tidlists(self) -> dict:
        assert self.dataset is not None
        tidlists = dict()

        for index, row in self.dataset.iterrows():
            for product in row['transactions']:
                tidlists.setdefault(product, []).append(index)

        return tidlists

    def read_and_convert_data(self, dataset_path: str, separator: str):
        # TODO raise exceptions
        df = pd.read_csv(dataset_path, names=['transactions'])
        # Converting each row to numpy int array
        df['transactions'] = df['transactions'].apply(lambda x: np.fromstring(x, dtype=np.int, sep=separator))
        self.dataset = df
        self.num_of_transactions = self.dataset.shape[0]

    @staticmethod
    def not_empty_subsets_generator(itemset: np.ndarray):
        assert isinstance(itemset, np.ndarray)

        length = len(itemset)
        for i in range(1, length):
            yield from combinations(itemset, r=i)

    def _association_rule_generator(self, itemset: np.ndarray):
        assert isinstance(itemset, np.ndarray)

        a_as_set = set(list(itemset))
        for antecedent in self.not_empty_subsets_generator(itemset):
            antecedent = set(antecedent)
            consequent = a_as_set - antecedent

            antecedent = np.sort(np.array(list(antecedent), dtype=int))
            consequent = np.sort(np.array(list(consequent), dtype=int))

            yield antecedent, consequent

    def association_rules_generator(self):
        assert self.frequent_itemsets is not None

        # not starting from 0 because association rules are only created from itemset of length of at least 2
        for i in range(1, len(self.frequent_itemsets)):
            for j in range(len(self.frequent_itemsets[i])):
                itememset = self.frequent_itemsets[i][j][0]
                for antecedent, consequent in self._association_rule_generator(itememset):
                    yield antecedent, consequent, itememset

    def frequent_association_rules_generator_with_metrics(
            self, calc_cosine: bool, calc_conviction: bool, calc_certainty_factor: bool):

        for antecedent, consequent, itemest in self.association_rules_generator():
            antecedent_sup = self.get_itemset_support(antecedent)
            consequent_sup = self.get_itemset_support(consequent)
            itemset_sup = self.get_itemset_support(itemest)

            confidence = self.calculate_confidence(
                antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup)
            cosine = None
            conviction = None
            certainty_factor = None

            if confidence > self._min_confidence:
                if calc_cosine:
                    cosine = self.calculate_cosine(
                        antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup)
                if calc_conviction:
                    conviction = self.calculate_conviction(
                        antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup, confidence)
                if calc_certainty_factor:
                    certainty_factor = self.calculate_certainty_factor(
                        antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup, confidence)

                yield antecedent, consequent, confidence, cosine, conviction, certainty_factor

    def get_itemset_support(self, itemset):
        return self.frequent_itemsets_index[np.array_str(itemset)]

    def calculate_confidence(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                             antecedent_sup: int = None, consequent_sup: int = None, itemset_sup: int = None):
        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        confidence = itemset_sup / float(antecedent_sup)

        return confidence

    def calculate_cosine(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                         antecedent_sup: int = None, consequent_sup: int = None, itemset_sup: int = None):
        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        cosine = itemset_sup / float(sqrt(antecedent_sup * consequent_sup))

        return cosine

    def calculate_conviction(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                             antecedent_sup: int = None, consequent_sup: int = None,
                             itemset_sup: int = None, confidence: float = None):

        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        if confidence is None:
            confidence = self.calculate_confidence(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)
        else:
            assert isinstance(confidence, float) and 0 <= confidence <= 1

        if confidence == 1:
            conviction = float('inf')
        else:
            conviction = (1 - consequent_sup) / float(1 - confidence)

        return conviction

    def calculate_certainty_factor(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                                   antecedent_sup: int = None, consequent_sup: int = None,
                                   itemset_sup: int = None, confidence: float = None):

        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        if confidence is None:
            confidence = self.calculate_confidence(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)
        else:
            assert isinstance(confidence, float) and 0 <= confidence <= 1

        certainty_factor = (confidence - consequent_sup) / (1 - consequent_sup)

        return certainty_factor

    def _check_itemsets_and_supports(self, antecedent: np.ndarray, consequent: np.ndarray,
                                     itemset: np.ndarray = None, antecedent_sup: int = None,
                                     consequent_sup: int = None, itemset_sup: int = None) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        assert isinstance(antecedent, np.ndarray)
        assert isinstance(consequent, np.ndarray)
        assert itemset is None or isinstance(itemset, np.ndarray)

        if antecedent_sup is None:
            antecedent_sup = self.get_itemset_support(antecedent)
        else:
            assert isinstance(antecedent_sup, float) and antecedent_sup > 0

        if consequent_sup is None:
            consequent_sup = self.get_itemset_support(consequent)
        else:
            assert isinstance(consequent_sup, float) and consequent_sup > 0

        if itemset_sup is None:
            if itemset is None:
                itemset = np.sort(np.concatenate((antecedent, consequent)))

            itemset_sup = self.get_itemset_support(itemset)
        else:
            if itemset_sup > consequent_sup or itemset_sup > antecedent_sup:
                raise ValueError('Given itemset support is bigger than antecedent or consequent support')

        return antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup

    def print_results_to_console(self):
        assert self.frequent_itemsets_index is not None
        frequent_association_rule_gen = self.frequent_association_rules_generator_with_metrics(
            calc_cosine=True, calc_conviction=True, calc_certainty_factor=True)

        for antecedent, consequent, confidence, cosine, conviction, certainty_factor in frequent_association_rule_gen:
            print("{} -> {}, confidence: {:.3f}, cosine: {:.3f}, conviction: {:.3f}, certainty_factor: {:.3f}".format(
                antecedent, consequent, confidence, cosine, conviction, certainty_factor))

    def _create_association_rules_dataframe(self):
        assert self.frequent_itemsets_index is not None

        antecedents = []
        consequents = []
        confidences = []
        cosines = []
        convictions = []
        certainty_factors = []

        frequent_association_rule_gen = self.frequent_association_rules_generator_with_metrics(
            calc_cosine=True, calc_conviction=True, calc_certainty_factor=True)
        for antecedent, consequent, confidence, cosine, conviction, certainty_factor in frequent_association_rule_gen:
            antecedents.append(antecedent)
            consequents.append(consequent)
            confidences.append(confidence)
            cosines.append(cosine)
            convictions.append(conviction)
            certainty_factors.append(certainty_factor)

        results_dict = {
            'antecedent': antecedents, 'consequent': consequents,
            'confidence': confidences, 'cosine': cosines,
            'conviction': convictions, 'certainty_factor': certainty_factors
        }
        results_dataframe = pd.DataFrame(results_dict)
        self.results_dataframe = results_dataframe

        return results_dataframe

    def get_results_dataframe(self) -> pd.DataFrame:
        if self.results_dataframe is None:
            self._create_association_rules_dataframe()

        return self.results_dataframe

    def save_results_to_csv(self, file_name: str):
        assert self.frequent_itemsets_index is not None
        assert isinstance(file_name, str)

        if self.results_dataframe is None:
            self._create_association_rules_dataframe()

        self.results_dataframe.to_csv(file_name, index=False)

    def save_results_to_json(self, file_name: str, orient: str = 'index'):
        assert self.frequent_itemsets_index is not None
        assert isinstance(file_name, str)

        if self.results_dataframe is None:
            self._create_association_rules_dataframe()

        self.results_dataframe.to_json(file_name, orient=orient, indent=4)

    def save_results_to_excel(self, file_name: str):
        assert self.frequent_itemsets_index is not None
        assert isinstance(file_name, str)

        if self.results_dataframe is None:
            self._create_association_rules_dataframe()

        self.results_dataframe.to_excel(file_name, index=False)
