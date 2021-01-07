import logging
from itertools import combinations
from math import sqrt
from typing import Tuple

import numpy as np
import pandas as pd

from .frequent_itemsets import Eclat


class AssociationRulesGenerator:
    ECLAT = 'ECLAT'

    def __init__(self, algorithm: str, dataset_path: str, min_support: float,
                 min_confidence: float, separator: str = ' ',
                 cosine: bool = True, certainty_f: bool = True, conviction: bool = True):
        assert isinstance(dataset_path, str)
        algorithms_lookup = {self.ECLAT: Eclat}
        algorithm = algorithms_lookup.get(algorithm, None)

        if algorithm is None:
            raise ValueError('Not know algorithm')

        self.algorithm = algorithm(dataset_path, min_support, separator)
        self.min_confidence = min_confidence

        self.frequent_itemsets = None
        self.frequent_itemsets_index = None
        self.results_dataframe = None

        self.calc_cosine = cosine
        self.calc_certainty_f = certainty_f
        self.calc_conviction = conviction

    @property
    def min_confidence(self):
        return self._min_confidence

    @min_confidence.setter
    def min_confidence(self, min_confidence: float):
        if not isinstance(min_confidence, float) or not 0 < min_confidence < 1:
            raise ValueError("Minimal confidence must be floating point number in interval (0, 1)")

        self._min_confidence = min_confidence

    def find_rules(self):
        logging.info('Finding frequent itemsets...')
        self.frequent_itemsets = self.algorithm.get_frequent_itemsets()
        logging.info('Indexing frequent itemsets...')
        self._create_frequent_itemsets_index()
        return self._create_association_rules_dataframe()

    def _create_frequent_itemsets_index(self) -> None:
        assert self.frequent_itemsets is not None
        index = {}
        for row in self.frequent_itemsets:
            for itemset_information in row:
                index[np.array_str(itemset_information[0])] = itemset_information[1]

        self.frequent_itemsets_index = index

    def _create_association_rules_dataframe(self):
        assert self.frequent_itemsets_index is not None
        logging.info('Creating association rules...')
        antecedents = []
        consequents = []
        confidences = []
        lifts = []
        cosines = []
        convictions = []
        certainty_factors = []

        frequent_association_rule_gen = self._frequent_association_rules_generator_with_metrics()
        for antecedent, consequent, confidence, lift, cosine, conviction, certainty_factor in frequent_association_rule_gen:
            antecedents.append(antecedent)
            consequents.append(consequent)
            confidences.append(confidence)
            lifts.append(lift)
            if self.calc_cosine:
                cosines.append(cosine)
            if self.calc_conviction:
                convictions.append(conviction)
            if self.calc_certainty_f:
                certainty_factors.append(certainty_factor)

        results_dict = {
            'antecedent': antecedents, 'consequent': consequents,
            'confidence': confidences, 'lift': lifts
        }
        if self.calc_cosine:
            results_dict['cosine'] = cosines
        if self.calc_conviction:
            results_dict['conviction'] = convictions
        if self.calc_certainty_f:
            results_dict['certainty_factor'] = certainty_factors

        results_dataframe = pd.DataFrame(results_dict)
        self.results_dataframe = results_dataframe
        logging.info('Association rules created...')
        return results_dataframe

    @staticmethod
    def _not_empty_subsets_generator(itemset: np.ndarray):
        assert isinstance(itemset, np.ndarray)

        length = len(itemset)
        for i in range(1, length):
            yield from combinations(itemset, r=i)

    def _frequent_association_rules_generator_with_metrics(self):

        for antecedent, consequent, itemest in self._association_rules_generator():
            antecedent_sup = self._get_itemset_support(antecedent)
            consequent_sup = self._get_itemset_support(consequent)
            itemset_sup = self._get_itemset_support(itemest)

            confidence = self._calculate_confidence(
                antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup)
            lift = self._calculate_lift(
                antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup, confidence)
            cosine = None
            conviction = None
            certainty_factor = None

            if confidence > self._min_confidence:
                if self.calc_cosine:
                    cosine = self._calculate_cosine(
                        antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup)
                if self.calc_conviction:
                    conviction = self._calculate_conviction(
                        antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup, confidence)
                if self.calc_certainty_f:
                    certainty_factor = self._calculate_certainty_factor(
                        antecedent, consequent, itemest, antecedent_sup, consequent_sup, itemset_sup, confidence)

                yield antecedent, consequent, confidence, lift, cosine, conviction, certainty_factor

    def _association_rules_generator(self):
        assert self.frequent_itemsets is not None

        # not starting from 0 because association rules are only created from itemset of length of at least 2
        for i in range(1, len(self.frequent_itemsets)):
            for j in range(len(self.frequent_itemsets[i])):
                itemset = self.frequent_itemsets[i][j][0]
                itemset_as_set = set(list(itemset))

                for antecedent in self._not_empty_subsets_generator(itemset):
                    antecedent = set(antecedent)
                    consequent = itemset_as_set - antecedent

                    antecedent = np.sort(np.array(list(antecedent), dtype=int))
                    consequent = np.sort(np.array(list(consequent), dtype=int))

                    yield antecedent, consequent, itemset

    def _get_itemset_support(self, itemset) -> int:
        return self.frequent_itemsets_index[np.array_str(itemset)]

    def _calculate_confidence(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                              antecedent_sup: int = None, consequent_sup: int = None, itemset_sup: int = None) -> float:
        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        confidence = itemset_sup / float(antecedent_sup)

        return confidence

    def _calculate_cosine(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                          antecedent_sup: int = None, consequent_sup: int = None, itemset_sup: int = None) -> float:
        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        cosine = itemset_sup / float(sqrt(antecedent_sup * consequent_sup))

        return cosine

    def _calculate_lift(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                        antecedent_sup: int = None, consequent_sup: int = None,
                        itemset_sup: int = None, confidence: float = None) -> float:

        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        if confidence is None:
            confidence = self._calculate_confidence(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)
        else:
            assert isinstance(confidence, float) and 0 <= confidence <= 1

        lift = confidence / consequent_sup

        return lift

    def _calculate_conviction(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                              antecedent_sup: int = None, consequent_sup: int = None,
                              itemset_sup: int = None, confidence: float = None) -> float:

        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        if confidence is None:
            confidence = self._calculate_confidence(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)
        else:
            assert isinstance(confidence, float) and 0 <= confidence <= 1

        if confidence == 1:
            conviction = float('inf')
        else:
            conviction = (1 - consequent_sup) / float(1 - confidence)

        return conviction

    def _calculate_certainty_factor(self, antecedent: np.ndarray, consequent: np.ndarray, itemset: np.ndarray = None,
                                    antecedent_sup: int = None, consequent_sup: int = None,
                                    itemset_sup: int = None, confidence: float = None) -> float:

        assert self.frequent_itemsets_index is not None

        antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup = \
            self._check_itemsets_and_supports(
                antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup)

        if confidence is None:
            confidence = self._calculate_confidence(
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
            antecedent_sup = self._get_itemset_support(antecedent)
        else:
            assert isinstance(antecedent_sup, float) and antecedent_sup > 0

        if consequent_sup is None:
            consequent_sup = self._get_itemset_support(consequent)
        else:
            assert isinstance(consequent_sup, float) and consequent_sup > 0

        if itemset_sup is None:
            if itemset is None:
                itemset = np.sort(np.concatenate((antecedent, consequent)))

            itemset_sup = self._get_itemset_support(itemset)
        else:
            if itemset_sup > consequent_sup or itemset_sup > antecedent_sup:
                raise ValueError('Given itemset support is bigger than antecedent or consequent support')

        return antecedent, consequent, itemset, antecedent_sup, consequent_sup, itemset_sup

    def get_results_dataframe(self) -> pd.DataFrame:
        if self.results_dataframe is None:
            self._create_association_rules_dataframe()

        return self.results_dataframe

    def print_results_to_console(self):
        assert self.frequent_itemsets_index is not None
        frequent_association_rule_gen = self._frequent_association_rules_generator_with_metrics()

        for antecedent, consequent, confidence, lift, cosine, conviction, certainty_factor in frequent_association_rule_gen:
            print(
                "{} -> {}, confidence: {:.3f}, lift: {:.3f}, cosine: {:.3f}, conviction: {:.3f}, certainty_factor: {:.3f}".format(
                    antecedent, consequent, confidence, lift, cosine, conviction, certainty_factor))

    def save_results_to_csv(self, file_name: str):
        assert self.frequent_itemsets_index is not None
        assert isinstance(file_name, str)
        logging.info('Saving results to csv file...')

        if self.results_dataframe is None:
            self._create_association_rules_dataframe()

        self.results_dataframe.to_csv(file_name, index=False)

    def save_results_to_json(self, file_name: str, orient: str = 'index'):
        assert self.frequent_itemsets_index is not None
        assert isinstance(file_name, str)
        logging.info('Saving results to json file...')

        if self.results_dataframe is None:
            self._create_association_rules_dataframe()

        self.results_dataframe.to_json(file_name, orient=orient, indent=4)

    def save_results_to_excel(self, file_name: str):
        assert self.frequent_itemsets_index is not None
        assert isinstance(file_name, str)
        logging.info('Saving results to excel file...')

        if self.results_dataframe is None:
            self._create_association_rules_dataframe()

        self.results_dataframe.to_excel(file_name, index=False)
