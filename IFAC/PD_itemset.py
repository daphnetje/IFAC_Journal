from itertools import product, chain, combinations
import pandas as pd

class PD_itemset:
    def __init__(self, dict_notation):
        self.dict_notation = dict_notation
        self.frozenset_notation = self.convert_to_frozenset_notation()
        self.string_notation = self.convert_to_string_notation()
        self.sensitive_features = self.sens_features_to_string()

    def __str__(self):
        return str(self.dict_notation)

    def __repr__(self):
        return str(self.dict_notation)

    def __eq__(self, another):
        return hasattr(another, 'dict_notation') and self.dict_notation == another.dict_notation

    def __hash__(self):
        return hash(self.frozenset_notation)

    def convert_to_frozenset_notation(self):
        initial_set = set()
        for key, item in self.dict_notation.items():
            string_notation = key + " : " + item
            initial_set.add(string_notation)

        return frozenset(initial_set)

    def convert_to_string_notation(self):
        string_notation = ""
        index_counter = 0
        for key, item in self.dict_notation.items():
            string_notation += item
            if (index_counter != (len(self.dict_notation)-1)):
                string_notation += ", "
            index_counter += 1
        return string_notation


    def sens_features_to_string(self):
        string_notation = ""
        index_counter = 0
        for key, item in self.dict_notation.items():
            string_notation += key
            if (index_counter != (len(self.dict_notation) - 1)):
                string_notation += ", "
            index_counter += 1
        return string_notation


def generate_potentially_discriminated_itemsets(data, sensitive_attributes):
    unique_values_per_sens_attribute = dict()

    for sens_attribute in sensitive_attributes:
        unique_values_of_sens_attribute = pd.unique(data.descriptive_data[sens_attribute])
        unique_values_per_sens_attribute[sens_attribute] = unique_values_of_sens_attribute

    all_pd_itemsets = []

    # Iterate over all subsets of the keys
    for subset in all_subsets(unique_values_per_sens_attribute.keys()):
        # For each subset, generate the Cartesian product of the corresponding values
        for values in product(*(unique_values_per_sens_attribute[key] for key in subset)):
            # Create a dictionary for this specific combination of keys and values
            combination = dict(zip(subset, values))
            combination_as_pd_itemset = PD_itemset(combination)
            all_pd_itemsets.append(combination_as_pd_itemset)

    return all_pd_itemsets



def generate_only_intersectional_potentially_discriminated_itemsets(data, sensitive_attributes):
    unique_values_per_sens_attribute = dict()

    for sens_attribute in sensitive_attributes:
        unique_values_of_sens_attribute = pd.unique(data.descriptive_data[sens_attribute])
        unique_values_per_sens_attribute[sens_attribute] = unique_values_of_sens_attribute

    all_intersectional_pd_itemsets = []

    # Iterate over all subsets of the keys
    for subset in all_subsets(unique_values_per_sens_attribute.keys()):
        # For each subset, generate the Cartesian product of the corresponding values
        for values in product(*(unique_values_per_sens_attribute[key] for key in subset)):
            # Create a dictionary for this specific combination of keys and values
            combination = dict(zip(subset, values))
            if len(combination) == len(sensitive_attributes):
                combination_as_pd_itemset = PD_itemset(combination)
                all_intersectional_pd_itemsets.append(combination_as_pd_itemset)

    return all_intersectional_pd_itemsets



def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss) + 1)))



