from .BlackBoxClassifier import BlackBoxClassifier
from .PD_itemset import generate_potentially_discriminated_itemsets
from .Rule import get_instances_covered_by_rule_base, get_instances_covered_by_rule, remove_rules_that_are_subsets_from_other_rules, convert_to_apriori_format, initialize_rule, calculate_support_conf_slift_and_significance
from .Rule import Rule, RuleIDGenerator
from .PD_itemset import PD_itemset
from .Reject import create_uncertainty_based_reject, create_unfairness_based_reject
from .SituationTesting import SituationTesting
from copy import deepcopy
from apyori import apriori
import pandas as pd
import itertools

class IFAC_Unfair_Uncertain_Rejects:

    def __init__(self, coverage, fairness_weight, sensitive_attributes, reference_group_list, val1_ratio=0.1, val2_ratio=0.1, base_classifier="Random Forest", max_pvalue_slift=0.01, sit_test_k = 5, sit_test_t = 0.2):
        self.coverage = coverage
        self.fairness_weight = fairness_weight
        self.sensitive_attributes = sensitive_attributes
        self.reference_group_list = reference_group_list
        self.val1_ratio = val1_ratio
        self.val2_ratio = val2_ratio
        self.base_classifier = base_classifier
        self.max_pvalue_slift = max_pvalue_slift
        self.sit_test_k = sit_test_k
        self.sit_test_t = sit_test_t
        self.rule_id_generator = RuleIDGenerator()


    def fit(self, X):
        print("Setting up IFAC_Unfair_Uncertain_Rejects")
        # Generate potentially discriminated itemsets
        self.pd_itemsets = generate_potentially_discriminated_itemsets(X, self.sensitive_attributes)
        self.decision_attribute = X.decision_attribute
        self.positive_label = X.desirable_label
        self.negative_label = X.undesirable_label
        self.class_items = frozenset([X.decision_attribute + " : " + X.undesirable_label, X.decision_attribute + " : " + X.desirable_label])

        #Step 0: Split into train and two validation sets
        val1_n = int(self.val1_ratio * len(X.descriptive_data))
        val2_n = int(self.val2_ratio * len(X.descriptive_data))
        print("val1: ", val1_n)
        X_train_dataset, X_val1_dataset = X.split_into_train_test(val1_n)
        X_train_dataset, X_val2_dataset = X_train_dataset.split_into_train_test(val2_n)

        #Step 1: Train Black-Box Model
        self.BB = BlackBoxClassifier(self.base_classifier)
        self.BB.fit(X_train_dataset)

        #Step 2: Extract at-risk subgroups dict, each key is a potentially_discriminated itemset (can be intersectional!) and each value
        #is a list of rules that are problematic
        #self.reject_rules = self.give_quick_sets_of_rules_for_income_testing_purposes()
        val_1_data_with_preds = self.make_preds_for_data(X_val1_dataset)
        self.reject_rules = self.learn_reject_rules(val_1_data_with_preds)
        save_reject_rules_in_excel_file(self.reject_rules)
        self.print_all_reject_rules()

        #Step 3: Prepare situation testing
        val_1_data_with_preds_and_probas = self.make_preds_and_preds_proba_for_data(X_val1_dataset)
        val_1_data_with_preds_and_probas.to_excel("adult_val_with_pred.xlsx")

        self.situationTester = SituationTesting(k=self.sit_test_k, t=self.sit_test_t, reference_group_list=self.reference_group_list, decision_label=self.decision_attribute, desirable_label=self.positive_label, distance_function = X.distance_function)
        self.situationTester.fit(val_1_data_with_preds_and_probas)

        #Learn uncertainty reject thresholds
        val_2_data_with_preds_and_probas = self.make_preds_and_preds_proba_for_data(X_val2_dataset)
        self.unfair_and_certain_limit, self.fair_and_uncertain_limit = self.learn_reject_thresholds(val_2_data_with_preds_and_probas)
        print(self.unfair_and_certain_limit)
        print(self.fair_and_uncertain_limit)
        return

    def make_preds_for_data(self, data_set):
        pred_for_data = self.BB.predict(data_set)
        data_descriptive = data_set.descriptive_data

        data_with_preds = deepcopy(data_descriptive)
        data_with_preds = data_with_preds.drop(columns=[self.decision_attribute])
        data_with_preds[self.decision_attribute] = pred_for_data
        return data_with_preds

    def make_preds_and_preds_proba_for_data(self, data_set):
        pred_for_data, prediction_probs_for_data = self.BB.predict_with_proba(data_set)
        data_descriptive = data_set.descriptive_data

        data_with_preds = deepcopy(data_descriptive)
        data_with_preds = data_with_preds.drop(columns=[self.decision_attribute])
        data_with_preds[self.decision_attribute] = pred_for_data
        data_with_preds['pred. probability'] = prediction_probs_for_data
        return data_with_preds

    def learn_class_rules_associated_with_prot_itemsets(self, val_data_with_preds):
        disc_rules_per_prot_itemset = {}
        for prot_itemset in self.pd_itemsets:
            print("Learning rules for: " + str(prot_itemset))
            disc_rules_for_prot_itemset = self.extract_disc_rules_for_one_prot_itemset(prot_itemset, val_data_with_preds)
            disc_rules_per_prot_itemset[prot_itemset] = disc_rules_for_prot_itemset

        return disc_rules_per_prot_itemset


    def extract_disc_rules_for_one_prot_itemset(self, prot_itemset, val_data):
        data_belonging_to_prot_itemset = get_instances_covered_by_rule_base(prot_itemset.dict_notation, val_data)
        data_belonging_to_prot_itemset = data_belonging_to_prot_itemset.drop(columns=self.sensitive_attributes)

        data_apriori_format = convert_to_apriori_format(data_belonging_to_prot_itemset)

        all_rules = list(apriori(transactions=data_apriori_format, min_support=0.01,
                               min_confidence=0.7, min_lift=1.0, min_length=2,
                               max_length=4))

        discriminatory_rules = []

        for rule in all_rules:
            if rule.items.isdisjoint(self.class_items):
                continue
            for ordering in rule.ordered_statistics:
                rule_base = ordering.items_base
                rule_consequence = ordering.items_add
                if (not rule_consequence.isdisjoint(self.class_items)) & (len(rule_consequence) == 1):
                    rule_base_with_prot_itemset = rule_base.union(prot_itemset.frozenset_notation)
                    myRule = initialize_rule(rule_base_with_prot_itemset, rule_consequence, self.rule_id_generator)
                    support_over_all_data, conf_over_all_data, slift, slift_p = calculate_support_conf_slift_and_significance(
                        myRule, val_data, prot_itemset)
                    myRule.set_support(support_over_all_data); myRule.set_confidence(conf_over_all_data)
                    myRule.set_slift(slift); myRule.set_slift_p_value(slift_p)
                    discriminatory_rules.append(myRule)
        return discriminatory_rules

    def learn_reject_rules(self, val_data_with_preds):
        class_rules_per_prot_itemset = self.learn_class_rules_associated_with_prot_itemsets(val_data_with_preds)

        reject_rules_per_prot_itemset = {}
        for pd_itemset in self.pd_itemsets:
            if pd_itemset.dict_notation != {}:
                reject_rules_per_prot_itemset[pd_itemset] = []

        reject_rules_per_prot_itemset = {}
        #could do a more advanced thing here, also taking confidence and everything into account
        for pd_itemset, rules in class_rules_per_prot_itemset.items():
            if pd_itemset.dict_notation != {}:
                #significant_rules_with_high_slift = [rule for rule in rules if ((rule.slift > 0.4) & (rule.slift_p_value < self.max_pvalue_slift))]

                significant_rules_with_high_slift = [rule for rule in rules if (((rule.confidence - rule.slift) < 0.5) & (rule.slift_p_value < self.max_pvalue_slift))]
                rules_with_high_slift_no_subrules = remove_rules_that_are_subsets_from_other_rules(significant_rules_with_high_slift)
                reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_subrules
                #this part is written to only have 'favouritism' rules for reference group
                if pd_itemset not in self.reference_group_list:
                    rules_with_high_slift_no_subrules_no_favouritism_Rules = [rule for rule in rules_with_high_slift_no_subrules if (rule.rule_consequence[self.decision_attribute] != self.positive_label)]
                    reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_subrules_no_favouritism_Rules
                    # this part is written to not have any discriminatory rules for reference group
                if pd_itemset in self.reference_group_list:
                    rules_with_high_slift_no_subrules_no_disc_Rules = [rule for rule in rules_with_high_slift_no_subrules if (rule.rule_consequence[self.decision_attribute] != self.negative_label)]
                    reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_subrules_no_disc_Rules

        return reject_rules_per_prot_itemset

    def print_all_reject_rules(self):
        for pd_itemset, reject_rules in self.reject_rules.items():
            print(pd_itemset)
            for reject_rule in reject_rules:
                print(reject_rule)

    def learn_reject_thresholds(self, val_data_with_preds):
        #first need to understand which instances are covered by reject rules
        val_data_covered_by_rules, relevant_rules_per_index = self.extract_data_falling_under_rules(val_data_with_preds)
        #afterwards need to run situation testing
        sit_test_labels, sit_test_scores, sit_test_info = self.situationTester.predict_disc_labels(
            val_data_covered_by_rules)
        discriminated_indices = (sit_test_labels[sit_test_labels == True]).index

        unfair_proportion_of_predictions = val_data_with_preds.loc[discriminated_indices]
        fair_proportion_of_predictions = val_data_with_preds[~val_data_with_preds.index.isin(discriminated_indices)]

        n_total_rejections = int((1-self.coverage) * len(val_data_with_preds))
        n_unfair_rejections = min(int(n_total_rejections * self.fairness_weight), len(unfair_proportion_of_predictions))
        n_uncertainty_rejections = n_total_rejections - n_unfair_rejections

        t_unfair_data = self.decide_on_probability_threshold_unfair_but_certain(unfair_proportion_of_predictions, n_unfair_rejections)
        t_uncertain_data = self.decide_on_probability_threshold_fair_but_uncertain(fair_proportion_of_predictions, n_uncertainty_rejections)

        return t_unfair_data, t_uncertain_data


    def extract_data_falling_under_rules(self, data):
        reject_rules_as_list = list(itertools.chain.from_iterable(self.reject_rules.values()))

        data_covered_by_rules = pd.DataFrame([])
        relevant_rules_per_index = pd.Series([])

        relevant_data = deepcopy(data)

        for rule in reject_rules_as_list:
            data_covered_by_rule = get_instances_covered_by_rule(rule, relevant_data)
            indices_covered_by_rule = pd.Series(rule, index=data_covered_by_rule.index)
            data_covered_by_rules = pd.concat([data_covered_by_rules, data_covered_by_rule], axis=0)
            relevant_rules_per_index = pd.concat([relevant_rules_per_index, indices_covered_by_rule])
            #remove the data that is covered by one rule from rest of relevant data
            relevant_data = relevant_data.drop(data_covered_by_rule.index)

        return data_covered_by_rules, relevant_rules_per_index

    # Meaning of cut_off_probability: if an instance falls under a discriminatory rule and has a high disc score ->
    # Reject from making a prediciton if prob is BIGGER than cut_off_value (unfair but certain)
    # Else (if prob is SMALLER than cut_off_value) than Intervene (unfair and uncertain)
    def decide_on_probability_threshold_unfair_but_certain(self, relevant_data, n_instances_to_reject):
        prediction_probs_of_data = relevant_data['pred. probability']
        ordered_prediction_probs = prediction_probs_of_data.sort_values(ascending=False)

        if (n_instances_to_reject >= len(relevant_data)):
            cut_off_probability = 0.5

        else:
            cut_off_probability = ordered_prediction_probs.iloc[n_instances_to_reject-1]

        return cut_off_probability

    # Meaning of cut_off_probability: if an instance doesn't fall under any of the discrimination rules OR doesn't have a high
    # disc score, then we are only going to reject that instance if it's prediction_probability is SMALLER than the cut_off_probability
    def decide_on_probability_threshold_fair_but_uncertain(self, relevant_data, n_instances_to_reject):
        prediction_probs_of_data = relevant_data['pred. probability']
        ordered_prediction_probs = prediction_probs_of_data.sort_values(ascending=True)

        if (n_instances_to_reject == 0):
            cut_off_probability = 0.5

        elif (n_instances_to_reject > len(relevant_data)):
            cut_off_probability = 1.0

        else:
            cut_off_probability = ordered_prediction_probs.iloc[n_instances_to_reject-1]

        return cut_off_probability

    def predict(self, test_dataset):
        #Step 1: Apply black box classifier, and store predictions
        test_data_with_preds = self.make_preds_and_preds_proba_for_data(test_dataset)
        test_data_with_preds.to_excel("test_data_with_preds.xlsx")
        predictions = test_data_with_preds[self.decision_attribute]

        #Step 2: Check which instances fall under reject rules
        test_data_covered_by_rules, relevant_rule_per_index = self.extract_data_falling_under_rules(test_data_with_preds)

        #Step 3: Run situation testing on those instances
        sit_test_labels, sit_test_scores, sit_test_info = self.situationTester.predict_disc_labels(test_data_covered_by_rules)
        discriminated_indices = (sit_test_labels[sit_test_labels == True]).index

        #Step 4: Divide into fair + unfair counterpart
        unfair_proportion_of_predictions = test_data_with_preds.loc[discriminated_indices]
        fair_proportion_of_predictions = test_data_with_preds[~test_data_with_preds.index.isin(discriminated_indices)]

        #Step 5: Apply different reject thresholds on both parts
        to_reject_from_unfair_part = unfair_proportion_of_predictions[unfair_proportion_of_predictions['pred. probability'] >= self.unfair_and_certain_limit]
        to_flip_from_unfair_part = unfair_proportion_of_predictions[unfair_proportion_of_predictions['pred. probability'] < self.unfair_and_certain_limit]

        sit_test_info_rejected_instances = sit_test_info.loc[to_reject_from_unfair_part.index]
        relevant_rules_rejected_instances = relevant_rule_per_index.loc[to_reject_from_unfair_part.index]
        sit_test_info_flipped_instances = sit_test_info.loc[to_flip_from_unfair_part.index]
        relevant_rules_flipped_instances = relevant_rule_per_index.loc[to_flip_from_unfair_part.index]

        to_reject_from_fair_part = fair_proportion_of_predictions[fair_proportion_of_predictions['pred. probability'] <= self.fair_and_uncertain_limit]

        all_unfairness_based_rejects_df = pd.DataFrame({
            'instance': [to_reject_from_unfair_part.loc[i].to_dict() for i in to_reject_from_unfair_part.index],
            'prediction_without_reject':  to_reject_from_unfair_part[self.decision_attribute],
            'prediction probability':  to_reject_from_unfair_part['pred. probability'],
            'relevant_rule': relevant_rules_rejected_instances,
            'sit_test_info': sit_test_info_rejected_instances,
        }, index=to_reject_from_unfair_part.index)
        all_unfairness_based_rejects_series = all_unfairness_based_rejects_df.apply(create_unfairness_based_reject, axis=1)

        all_unfairness_based_flips_df = pd.DataFrame({
            'instance': [to_flip_from_unfair_part.loc[i].to_dict() for i in to_flip_from_unfair_part.index],
            'prediction_without_reject': to_flip_from_unfair_part[self.decision_attribute],
            'prediction probability': to_flip_from_unfair_part['pred. probability'],
            'relevant_rule': relevant_rules_flipped_instances,
            'sit_test_info': sit_test_info_flipped_instances,
        }, index=to_flip_from_unfair_part.index)
        all_unfairness_based_flips_series = all_unfairness_based_flips_df.apply(create_unfairness_based_reject, axis=1)
        org_predictions_to_be_flipped = to_flip_from_unfair_part[self.decision_attribute]
        flipped_predictions = org_predictions_to_be_flipped.replace({self.negative_label: self.positive_label, self.positive_label: self.negative_label})

        all_uncertainty_based_rejects_df = pd.DataFrame({
            'instance': [to_reject_from_fair_part.loc[i].to_dict() for i in to_reject_from_fair_part.index],
            'prediction_without_reject': to_reject_from_fair_part[self.decision_attribute],
            'prediction probability': to_reject_from_fair_part['pred. probability'],
        }, index=to_reject_from_fair_part.index)

        all_uncertainty_based_rejects_series = (pd.Series([]) if all_uncertainty_based_rejects_df.empty
            else all_uncertainty_based_rejects_df.apply(create_uncertainty_based_reject, axis=1))

        print("ORG IFAC is rejecting " + str(len(all_unfairness_based_rejects_series) + len(all_uncertainty_based_rejects_series)) + " instances")

        predictions.update(all_unfairness_based_rejects_series)
        predictions.update(all_uncertainty_based_rejects_series)
        predictions.update(flipped_predictions)
        return predictions, all_unfairness_based_flips_series

    def give_quick_sets_of_rules_for_income_testing_purposes(self):
        disc_class_rules_connected_to_pd_itemsets = dict()
        for pd_itemset in self.pd_itemsets:
            disc_class_rules_connected_to_pd_itemsets[pd_itemset] = []

        female_and_white = PD_itemset({"sex": "Female", "race": "White alone"})
        rule_1 = Rule({"sex": "Female", "race": "White alone", "marital status": "Married"}, {"income": "low"},
                      support=0.02, confidence=0.9, lift=1.0, slift=0.51, slift_p_value=0.00)
        rule_2 = Rule({"sex": "Female", "race": "White alone", "workinghours": "More than 50"}, {"income": "low"},
                      support=0.02, confidence=0.9, lift=1.0, slift=0.61, slift_p_value=0.00)

        male_and_white = PD_itemset({"sex": "Male", "race": "White alone"})
        rule_3 = Rule({"sex": "Male", "race": "White alone", "workinghours": "More than 50"}, {"income": "high"},
                      support=0.02, confidence=0.9, lift=1.0, slift=0.51, slift_p_value=0.00)
        #
        male_and_other = PD_itemset({"sex": "Male", "race": "Other"})
        rule_4 = Rule({"sex": "Male", "race": "Other", "workinghours": "More than 50"}, {"income": "high"},
                      support=0.02, confidence=0.9, lift=1.0, slift=0.51, slift_p_value=0.00)
        rule_5 = Rule({"sex": "Male", "race": "Other", "marital status": "Seperated"}, {"income": "low"}, support=0.02,
                      confidence=0.9, lift=1.0, slift=0.51, slift_p_value=0.00)
        #
        male = PD_itemset({"sex": "Male"})
        rule_6 = Rule({"sex": "Male", "workinghours": "More than 50"}, {"income": "high"}, support=0.02, confidence=0.9,
                      lift=1.0, slift=0.51, slift_p_value=0.00)

        disc_class_rules_connected_to_pd_itemsets[female_and_white] = [rule_1, rule_2]
        disc_class_rules_connected_to_pd_itemsets[male_and_white] = [rule_3]
        disc_class_rules_connected_to_pd_itemsets[male_and_other] = [rule_4, rule_5]
        disc_class_rules_connected_to_pd_itemsets[male] = [rule_6]

        return disc_class_rules_connected_to_pd_itemsets



def save_reject_rules_in_excel_file(reject_rules):
    # Convert data_dict to a list of dictionaries
    data_rows = []
    for pd_itemset, rules in reject_rules.items():
        for rule in rules:
            rule_base_without_protected_itemset = deepcopy(rule.rule_base)
            for key in pd_itemset.dict_notation.keys():
                rule_base_without_protected_itemset.pop(key, None)

            data_rows.append({
                "pd_itemset": str(pd_itemset),
                "rule_base": rule_base_without_protected_itemset,
                "rule_conclusion": rule.rule_consequence,
                "support": rule.support,
                "confidence": rule.confidence,
                "slift": rule.slift,
                "p_value_slift": rule.slift_p_value
            })

    # Create a DataFrame
    df = pd.DataFrame(data_rows)

    # Save to an Excel file
    df.to_excel("reject_rules.xlsx", index=False, engine="openpyxl")