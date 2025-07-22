from copy import deepcopy
from scipy import stats
from math import sqrt

class Rule:
    def __init__(self, id, rule_base={}, rule_consequence={}, support=0, confidence=0, lift=0, slift=0, slift_p_value=0):
        self.id = id
        self.rule_base = rule_base
        self.rule_consequence = rule_consequence
        self.support = support
        self.confidence = confidence
        self.lift = lift
        self.slift = slift
        self.slift_p_value = slift_p_value

    def set_support(self, support):
        self.support = support

    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_slift(self, slift):
        self.slift = slift

    def set_slift_p_value(self, p_value):
        self.slift_p_value = p_value

    def __str__(self):
        output_string = "("
        counter = 1
        for rule_key in self.rule_base.keys():
            output_string += rule_key + " = " + str(self.rule_base[rule_key])
            if counter != len(self.rule_base):
                output_string += " AND "
            counter += 1
        output_string += ") -> "

        counter = 1
        for rule_key in self.rule_consequence.keys():
            output_string += "(" + rule_key + " = " + str(self.rule_consequence[rule_key])
            if counter != len(self.rule_consequence):
                output_string += " AND "
            counter += 1
        output_string += ")"

        output_string += f", Support: {self.support:.3f}, Confidence: {self.confidence:.3f}, Lift: {self.lift:.3f}, SLift: {self.slift:.3f}"
        return output_string

    def __repr__(self):
        output_string = "Rule: ("
        counter = 1
        for rule_key in self.rule_base.keys():
            output_string += self.rule_base[rule_key]
            if counter != len(self.rule_base):
                output_string += " AND "
            counter += 1
        output_string += ") -> "

        counter = 1
        for rule_key in self.rule_consequence.keys():
            output_string += str(self.rule_consequence[rule_key])
            if counter != len(self.rule_consequence):
                output_string += " AND "
            counter += 1

        output_string += f" Confidence: {self.confidence}, Support: {self.support}, Slift: {self.slift}, p-value: {self.slift_p_value}"
        return output_string


def get_instances_covered_by_rule_base(rule_base, data):
    relevant_data = data
    for key in rule_base.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_base[key]]
    return relevant_data


def get_instances_covered_by_rule(rule, data):
    relevant_data = data
    for key in rule.rule_base.keys():
        relevant_data = relevant_data[relevant_data[key] == rule.rule_base[key]]

    for key in rule.rule_consequence.keys():
        relevant_data = relevant_data[relevant_data[key] == rule.rule_consequence[key]]

    return relevant_data

def convert_frozenset_rule_format_to_dict_format(frozentset_rule_representation):
    rule_as_dict = {}
    #rule is a frozenset, where each item follows the following format 'key : value'
    #each itemstring needs to be added to the dictionary as one key, value pair
    for rule_item in frozentset_rule_representation:
        #rule_item is string
        splitted_rule = rule_item.split(" : ")
        key_of_rule = splitted_rule[0]
        value_of_rule = splitted_rule[1]
        if value_of_rule.isdigit():
            rule_as_dict[key_of_rule] = int(value_of_rule)
        else:
            rule_as_dict[key_of_rule] = value_of_rule
    return rule_as_dict

def initialize_rule(rule_base_frozenset, rule_consequence_frozenset, rule_id_generator):
    rule_base_dict = convert_frozenset_rule_format_to_dict_format(rule_base_frozenset)
    rule_consequence_dict = convert_frozenset_rule_format_to_dict_format(rule_consequence_frozenset)
    id = rule_id_generator.generate_rule_id()
    rule = Rule(id, rule_base_dict, rule_consequence_dict)
    return rule

def convert_to_apriori_format(X):
    list_of_dicts_format = X.to_dict('records')
    list_of_lists = []
    for dictionary in list_of_dicts_format:
        one_entry = set()
        for key, value in dictionary.items():
            one_entry.add(key + " : " + str(value))
        list_of_lists.append(one_entry)
    return list_of_lists


#rule come in this format {'rule_base': {'sex': 'Male'}, 'rule_consequence': {'income': '<=50K'}, 'support': 0.46460489542704464, 'confidence': 0.6942634235888022, 'lift': 0.9144786138946193}
def calculate_support_conf_slift_and_significance(rule, data, protected_itemset):
    pd_itemset_dict_notation = protected_itemset.dict_notation
    pd_itemset_frozenset_notation = protected_itemset.frozenset_notation

    #check if class rule contains protected itemset. If not than the DCI score will be 0
    if pd_itemset_frozenset_notation==frozenset():
        return 0, 0, 0

    n_covered_by_rule_base, n_covered_by_complete_rule = get_number_of_instances_covered_by_ruleBase_and_by_completeRule(rule.rule_base, rule.rule_consequence, data)
    confidence_org_rule = n_covered_by_complete_rule / n_covered_by_rule_base
    support_org_rule = n_covered_by_complete_rule / len(data)

    rule_base_without_protected_itemset = deepcopy(rule.rule_base)
    for key in pd_itemset_dict_notation.keys():
        rule_base_without_protected_itemset.pop(key, None)

    n_covered_by_rule_base_with_neg_prot_itemset, n_covered_by_complete_rule_with_neg_prot_itemset = get_number_of_instances_covered_by_ruleBase_and_completeRule_with_neg_part(
        rule_base_without_protected_itemset, protected_itemset.dict_notation, rule.rule_consequence, data)
    if n_covered_by_rule_base_with_neg_prot_itemset != 0:
        confidence_org_rule_neg_prot_itemset = n_covered_by_complete_rule_with_neg_prot_itemset/n_covered_by_rule_base_with_neg_prot_itemset
        slift_d = confidence_org_rule - confidence_org_rule_neg_prot_itemset
        p_value_slift_d = calculate_significance_of_slift(n_covered_by_rule_base,
                                                          n_covered_by_rule_base_with_neg_prot_itemset,
                                                          n_covered_by_complete_rule,
                                                          n_covered_by_complete_rule_with_neg_prot_itemset, len(data))
    else:
        p_value_slift_d = -999
        slift_d = -999

    return support_org_rule, confidence_org_rule, slift_d, p_value_slift_d


def get_number_of_instances_covered_by_ruleBase_and_by_completeRule(rule_base, rule_consequence, data):
    relevant_data = data
    for key in rule_base.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_base[key]]

    number_of_instances_covered_by_rule_base = len(relevant_data)

    for key in rule_consequence.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_consequence[key]]

    number_of_instances_covered_by_rule_base_and_consequence = len(relevant_data)
    return number_of_instances_covered_by_rule_base, number_of_instances_covered_by_rule_base_and_consequence


def get_instances_covered_by_rule_with_negation(rule_base, negation_part, data):
    non_relevant_data = data

    for key in negation_part.keys():
        non_relevant_data = non_relevant_data[non_relevant_data[key] == negation_part[key]]

    index_relevance_boolean_indicators = data.index.isin(non_relevant_data.index)
    relevant_data = data[~index_relevance_boolean_indicators]

    for key in rule_base.keys():
        relevant_data = relevant_data[relevant_data[key] == rule_base[key]]

    return relevant_data


def get_number_of_instances_covered_by_ruleBase_and_completeRule_with_neg_part(rule_base, negation_part_rule_base, rule_consequence, data):
    instances_covered_by_rule_base_with_negation = get_instances_covered_by_rule_with_negation(rule_base,
                                                                                               negation_part_rule_base,
                                                                                               data)
    n_instances_covered_by_rule_base_with_negation = len(instances_covered_by_rule_base_with_negation)

    instances_covered_by_rule_base_with_negation_and_consequence = instances_covered_by_rule_base_with_negation

    for key in rule_consequence.keys():
        instances_covered_by_rule_base_with_negation_and_consequence = \
        instances_covered_by_rule_base_with_negation_and_consequence[
            instances_covered_by_rule_base_with_negation_and_consequence[key] == rule_consequence[key]]

    n_instances_covered_by_rule_base_with_negation_and_consequence = len(
        instances_covered_by_rule_base_with_negation_and_consequence)

    return n_instances_covered_by_rule_base_with_negation, n_instances_covered_by_rule_base_with_negation_and_consequence

def calculate_significance_of_slift(number_instances_covered_by_org_rule_base, number_instances_covered_by_ref_rule_base, number_instances_covered_by_complete_org_rule, number_instances_covered_by_complete_ref_rule, total_number_instances):
    confidence_org_rule = number_instances_covered_by_complete_org_rule / number_instances_covered_by_org_rule_base
    confidence_reference_pd_rule = number_instances_covered_by_complete_ref_rule / number_instances_covered_by_ref_rule_base

    slift_d= confidence_org_rule - confidence_reference_pd_rule
    if (slift_d == 0):
        p_value = 1.0
        return p_value

    if (confidence_reference_pd_rule == 0):
        p_value = 0.0
        return p_value

    total_proportion_both_groups= (number_instances_covered_by_complete_org_rule + number_instances_covered_by_complete_ref_rule) / (number_instances_covered_by_org_rule_base + number_instances_covered_by_ref_rule_base)

    Z = (confidence_org_rule-confidence_reference_pd_rule) / sqrt((total_proportion_both_groups * (1 - total_proportion_both_groups) * ((1 / number_instances_covered_by_complete_org_rule) + (1 / number_instances_covered_by_complete_ref_rule))))

    p_value = stats.norm.sf(abs(Z))*2
    return p_value


def rule1_is_subset_of_rule2(rule1, rule2):
    if rule1.rule_consequence != rule2.rule_consequence:
        return False

    return set(rule2.rule_base.items()).issubset(set(rule1.rule_base.items()))

def remove_rules_that_are_subsets_from_other_rules(list_of_rules):
    result = []
    for index1, rule1 in enumerate(list_of_rules):
        is_subset_of_any = False
        for index2, rule2 in enumerate(list_of_rules):
            if index1 != index2 and rule1_is_subset_of_rule2(rule1, rule2):
                is_subset_of_any = True
                break
        if not is_subset_of_any:
            result.append(rule1)
    return result



class RuleIDGenerator():
    def __init__(self, start=1):
        self.counter = start

    def generate_rule_id(self):
        # Increment the counter and return the ID
        self.counter += 1
        return f"RULE{self.counter}"

