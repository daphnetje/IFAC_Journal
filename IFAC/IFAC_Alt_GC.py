'''
'''

from .BlackBoxClassifier import BlackBoxClassifier
from .PD_itemset import generate_potentially_discriminated_itemsets, generate_only_intersectional_potentially_discriminated_itemsets
from .Rule import get_instances_covered_by_rule_base, get_instances_covered_by_rule, remove_rules_that_are_subsets_from_other_rules, convert_to_apriori_format, initialize_rule, calculate_support_conf_slift_and_significance
from .Rule import Rule, RuleIDGenerator
from .PD_itemset import PD_itemset
from .Reject import create_uncertainty_based_reject, create_unfairness_based_reject, create_simple_reject, create_unfairness_based_flip
from .SituationTesting import SituationTesting, SituationTestingInfo
from copy import deepcopy
from apyori import apriori
import pandas as pd
import itertools
import numpy as np
import math
from random import uniform
from copulas.multivariate import GaussianMultivariate, VineCopula
from copulas.univariate import GaussianUnivariate
from copulas.datasets import sample_trivariate_xyz
import pyvinecopulib as pv

class IFAC_Alt_GC:

    def __init__(self, coverage, task, val1_ratio=0.1, val2_ratio=0.1, base_classifier="Random Forest", max_pvalue_slift=0.01, sit_test_k = 10):
        self.coverage = coverage
        self.task = task
        self.val1_ratio = val1_ratio
        self.val2_ratio = val2_ratio
        self.base_classifier = base_classifier
        self.max_pvalue_slift = max_pvalue_slift
        self.sit_test_k = sit_test_k
        self.rule_id_generator = RuleIDGenerator()


    def fit(self, X):
        self.sensitive_attributes = X.sensitive_attributes
        self.reference_group_list = [PD_itemset(X.reference_group_dict)]
        print("Setting up IFAC Rejects")
        # Generate potentially discriminated itemsets
        #self.pd_itemsets = generate_potentially_discriminated_itemsets(X, self.sensitive_attributes)
        self.pd_itemsets = generate_only_intersectional_potentially_discriminated_itemsets(X, self.sensitive_attributes)
        self.intersectional_pd_itemsets = generate_only_intersectional_potentially_discriminated_itemsets(X, self.sensitive_attributes)
        self.decision_attribute = X.decision_attribute
        self.positive_label = X.desirable_label
        self.negative_label = X.undesirable_label
        self.class_items = frozenset([X.decision_attribute + " : " + X.undesirable_label, X.decision_attribute + " : " + X.desirable_label])

        #Step 0: Split into train and two validation sets
        val1_n = int(self.val1_ratio * len(X.descriptive_data))
        val2_n = int(self.val2_ratio * len(X.descriptive_data))
        X_train_dataset, X_val1_dataset = X.split_into_train_test(val1_n)
        X_train_dataset, X_val2_dataset = X_train_dataset.split_into_train_test(val2_n)

        #Step 1: Train Black-Box Model
        self.BB = BlackBoxClassifier(self.base_classifier)
        self.BB.fit(X_train_dataset)

        #Step 2: Extract at-risk subgroups dict, each key is a potentially_discriminated itemset (can be intersectional!) and each value
        #is a list of rules that are problematic
        #self.reject_rules = self.give_quick_sets_of_rules_for_income_testing_purposes()
        val_1_data_with_preds = self.make_bb_preds_for_data(X_val1_dataset)
        self.reject_rules = self.learn_reject_rules(val_1_data_with_preds)
        save_reject_rules_in_excel_file(self.task, self.reject_rules)
        self.print_all_reject_rules()
        #
        #Step 3: Prepare situation testing
        val_1_data_with_preds_and_probas = self.make_bb_preds_and_preds_proba_for_data(X_val1_dataset)
        val_1_data_with_preds_and_probas.to_excel(self.task+"VAL1_DATA.xlsx")
        self.situationTester = SituationTesting(k=self.sit_test_k, t=0, reference_group_list=self.reference_group_list, decision_label=self.decision_attribute, desirable_label=self.positive_label, distance_function = X.distance_function)
        val_1_data_with_dummy_proba = deepcopy(X_val1_dataset.descriptive_data)
        val_1_data_with_dummy_proba['pred. probability'] = -1
        self.situationTester.fit(val_1_data_with_dummy_proba)
        #
        #
        # #For each group: calculate amount of instances that would need to be rejected to get equal pos-decision-raito over groups
        # #Learn uncertainty reject thresholds
        val_2_data_with_labels = X_val2_dataset.descriptive_data
        val_2_data_with_preds_and_probas = self.make_bb_preds_and_preds_proba_for_data(X_val2_dataset)

        self.copula_per_group, self.flip_thresholds_per_group, self.reject_threshold_per_group,  self.unc_neg_threshold_per_group, self.unc_pos_threshold_per_group = self.learn_reject_threshold(val_2_data_with_labels, val_2_data_with_preds_and_probas)
        return X_val1_dataset.descriptive_data

    def make_bb_preds_for_data(self, data_set):
        pred_for_data = self.BB.predict(data_set)
        data_descriptive = data_set.descriptive_data

        data_with_preds = deepcopy(data_descriptive)
        data_with_preds = data_with_preds.drop(columns=[self.decision_attribute])
        data_with_preds[self.decision_attribute] = pred_for_data
        return data_with_preds

    def make_bb_preds_and_preds_proba_for_data(self, data_set):
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
            disc_rules_for_prot_itemset = self.extract_disc_rules_for_one_prot_itemset(prot_itemset, val_data_with_preds)
            disc_rules_per_prot_itemset[prot_itemset] = disc_rules_for_prot_itemset

        return disc_rules_per_prot_itemset


    def extract_disc_rules_for_one_prot_itemset(self, prot_itemset, val_data):
        data_belonging_to_prot_itemset = get_instances_covered_by_rule_base(prot_itemset.dict_notation, val_data)
        data_belonging_to_prot_itemset = data_belonging_to_prot_itemset.drop(columns=self.sensitive_attributes)

        data_apriori_format = convert_to_apriori_format(data_belonging_to_prot_itemset)

        #min_confidence=0.85, min_lift=1.0
        all_rules = list(apriori(transactions=data_apriori_format, min_support=0.01,
                               min_confidence=0.7, min_lift=1, min_length=2,
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

        for pd_itemset, rules in class_rules_per_prot_itemset.items():
            if pd_itemset.dict_notation != {}:
                significant_rules_with_high_slift = [rule for rule in rules if ((rule.slift > 0.00) & (rule.slift_p_value < self.max_pvalue_slift))]
                reject_rules_per_prot_itemset[pd_itemset] = significant_rules_with_high_slift

                #this part is written to only have 'favouritism' rules for reference group
                if pd_itemset not in self.reference_group_list:
                    rules_with_high_slift_no_favouritism_Rules = [rule for rule in significant_rules_with_high_slift if (rule.rule_consequence[self.decision_attribute] != self.positive_label)]
                    reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_favouritism_Rules

                # this part is written to not have any discriminatory rules for reference group
                if pd_itemset in self.reference_group_list:
                    rules_with_high_slift_no_disc_Rules = [rule for rule in significant_rules_with_high_slift if (rule.rule_consequence[self.decision_attribute] != self.negative_label)]
                    reject_rules_per_prot_itemset[pd_itemset] = rules_with_high_slift_no_disc_Rules

        return reject_rules_per_prot_itemset

    def print_all_reject_rules(self):
        for pd_itemset, reject_rules in self.reject_rules.items():
            print(pd_itemset)
            for reject_rule in reject_rules:
                print(reject_rule)

    def learn_reject_threshold(self, val_data_with_labels, val_data_with_preds):
        n_to_flip_per_pd_itemset, n_to_reject_per_pd_itemset, n_uncertain_neg_reject_per_pd_itemset, n_uncertain_pos_reject_per_pd_itemset = \
            self.calculate_flip_and_reject_rates_per_group(val_data_with_labels, val_data_with_preds)

        # first need to understand which instances are covered by reject rules
        # Should extract HIGHEST slift rule here that covers an instance, not just any rule
        val_data_covered_by_rules, relevant_rules_per_index, slift_per_index = self.extract_highest_slift_rule_per_instance(
            val_data_with_preds)

        # afterwards need to run situation testing
        sit_test_labels, sit_test_scores, sit_test_info = self.situationTester.predict_disc_labels(
            val_data_covered_by_rules)

        #fill slift of instances that are not covered by rule with 0
        slift_per_index = slift_per_index.reindex(val_data_with_preds.index, fill_value=0)
        sit_test_scores = sit_test_scores.reindex(val_data_with_preds.index, fill_value=0)

        unf_flip_thresholds_per_intersectional_group = {}
        unf_reject_thresholds_per_intersectional_group = {}
        copula_per_intersectional_group = {}
        unc_pos_threshold_per_intersectional_group = {}
        unc_neg_threshold_per_intersectional_group = {}

        for pd_itemset in self.intersectional_pd_itemsets:
            data_from_itemset = get_instances_covered_by_rule_base(pd_itemset.dict_notation, val_data_with_preds)
            neg_data_from_itemset = data_from_itemset[
                data_from_itemset[self.decision_attribute] == self.negative_label]
            pos_data_from_itemset = data_from_itemset[
                data_from_itemset[self.decision_attribute] == self.positive_label]

            if pd_itemset in self.reference_group_list:
                unfair_part_of_data = pos_data_from_itemset
                fair_part_of_data = neg_data_from_itemset
            else:
                unfair_part_of_data = neg_data_from_itemset
                fair_part_of_data = pos_data_from_itemset

            #Learn the reject and flip thresholds for the unfair part of the data (i.e. positive prediction data in case of our reference group,
            #negative prediction data in case of our non_reference group)
            copula, flip_threshold, reject_threshold, non_rejected_data = self.learn_reject_and_flip_thresholds_unfair_data(unfair_part_of_data, slift_per_index, sit_test_scores, n_to_reject_per_pd_itemset[pd_itemset], n_to_flip_per_pd_itemset[pd_itemset])
            copula_per_intersectional_group[pd_itemset] = copula
            unf_flip_thresholds_per_intersectional_group[pd_itemset] = flip_threshold
            unf_reject_thresholds_per_intersectional_group[pd_itemset] = reject_threshold

            #Learn for all the remaining predictions, what are the thresholds for rejecting uncertaing negative predictins and uncertain positive ones
            unc_neg_threshold, unc_pos_threshold = self.learn_reject_thresholds_for_uncertain_data(fair_part_of_data, non_rejected_data,
                                                            n_uncertain_neg_reject_per_pd_itemset[pd_itemset]
                                                            , n_uncertain_pos_reject_per_pd_itemset[pd_itemset])
            unc_pos_threshold_per_intersectional_group[pd_itemset] = unc_pos_threshold
            unc_neg_threshold_per_intersectional_group[pd_itemset] = unc_neg_threshold

        return copula_per_intersectional_group, unf_flip_thresholds_per_intersectional_group, unf_reject_thresholds_per_intersectional_group, unc_neg_threshold_per_intersectional_group, unc_pos_threshold_per_intersectional_group


    def learn_reject_and_flip_thresholds_unfair_data(self, unfair_part_of_data, slift_per_index, sit_test_scores, n_to_reject, n_to_flip):
        slifts_for_itemset = slift_per_index.loc[unfair_part_of_data.index]
        print(slifts_for_itemset)
        print(unfair_part_of_data)
        sit_test_scores_for_itemset = sit_test_scores.loc[unfair_part_of_data.index]
        uncertainty_scores_for_itemset = 1 - unfair_part_of_data['pred. probability']

        copula, joint_cdf_scores, _, _ = self.fit_gaussian_copula(slifts_for_itemset, sit_test_scores_for_itemset, uncertainty_scores_for_itemset)

        joint_cdf_scores = joint_cdf_scores + np.random.uniform(-0.001, 0.001, size=len(unfair_part_of_data))
        print(joint_cdf_scores)

        flip_threshold, reject_threshold = self.decide_on_flip_and_reject_threshold(joint_cdf_scores,
                                                                                    n_to_flip,
                                                                                    n_to_reject)

        print("Number of unfairness based rejected instances: ", len(joint_cdf_scores[(joint_cdf_scores >= reject_threshold) & (joint_cdf_scores < flip_threshold)]))
        non_rejected_indices = joint_cdf_scores[joint_cdf_scores < reject_threshold].index
        non_rejected_data = unfair_part_of_data.loc[non_rejected_indices]
        return copula, flip_threshold, reject_threshold, non_rejected_data


    def fit_gaussian_copula(self, slifts, sit_scores, unc_scores):
        glu_dataset = pd.concat({'global_unf': slifts,
              'local_unf': sit_scores,
              'unc': unc_scores},axis=1)

        gl_dataset = pd.concat({'global_unf': slifts,
              'local_unf': sit_scores},axis=1)

        glu_dataset_transformed = pv.to_pseudo_obs(glu_dataset.to_numpy())

        controls = pv.FitControlsVinecop(family_set=[pv.gaussian])
        copula = pv.Vinecop.from_data(glu_dataset_transformed, controls=controls)
        cdf = pd.Series(glu_dataset_transformed, index=glu_dataset.index)
        glu_dataset['cdf'] = cdf
        print(glu_dataset)

        copula_with_unc = VineCopula('center')
        copula_with_unc.fit(glu_dataset)

        copula_just_fairness = VineCopula('center')
        copula_just_fairness.fit(glu_dataset)

        # compute joint CDF
        glu_cdf_score = pd.Series(copula_with_unc.cumulative_distribution(glu_dataset), index=glu_dataset.index)
        gl_cdf_score = pd.Series(copula_just_fairness.cumulative_distribution(gl_dataset), index=gl_dataset.index)

        return copula_with_unc, glu_cdf_score, copula_just_fairness, gl_cdf_score



    def learn_reject_thresholds_for_uncertain_data(self, fair_part_of_data, non_rejected_data, n_neg_uncertain_reject, n_pos_uncertain_reject):
        all_non_rejected_data = pd.concat([fair_part_of_data, non_rejected_data])
        neg_preds_uncertainty = 1 - all_non_rejected_data[
            all_non_rejected_data[self.decision_attribute] == self.negative_label]['pred. probability']
        pos_preds_uncertainty = 1 - all_non_rejected_data[
            all_non_rejected_data[self.decision_attribute] == self.positive_label]['pred. probability']
        unc_neg_threshold = self.decide_on_reject_threshold_uncertain_data(
            neg_preds_uncertainty, n_neg_uncertain_reject)
        unc_pos_threshold = self.decide_on_reject_threshold_uncertain_data(
            pos_preds_uncertainty, n_pos_uncertain_reject)
        return unc_neg_threshold, unc_pos_threshold

    def extract_highest_slift_rule_per_instance(self, data):
        reject_rules_as_list = list(itertools.chain.from_iterable(self.reject_rules.values()))

        coverage_records = []

        for rule in reject_rules_as_list:
            data_covered_by_rule = get_instances_covered_by_rule(rule, data)

            coverage_records.append(pd.DataFrame({
                'index': data_covered_by_rule.index,
                'rule': rule,
                'slift': rule.slift
            }))

        # Combine all coverage records into a single DataFrame
        coverage_df = pd.concat(coverage_records, ignore_index=True)

        # For each instance, find the pattern with the highest score
        highest_slift_df = coverage_df.loc[coverage_df.groupby('index')['slift'].idxmax()]

        # Create a pandas Series for the final result
        highest_slift_rule_per_instance = pd.Series(
            data=highest_slift_df['rule'].values,
            index=highest_slift_df['index']
        )

        highest_slift_per_instance = pd.Series(
            data=highest_slift_df['slift'].values,
            index=highest_slift_df['index']
        )

        data_covered_by_rules = data.loc[highest_slift_rule_per_instance.index]

        return data_covered_by_rules, highest_slift_rule_per_instance, highest_slift_per_instance

    # Meaning of cut_off_probability: if an instance falls under a discriminatory rule and has a high disc score ->
    # Reject from making a prediciton if prob is BIGGER than cut_off_value (unfair but certain)
    # Else (if prob is SMALLER than cut_off_value) than Intervene (unfair and uncertain)
    def decide_on_flip_and_reject_threshold(self, disc_scores, n_instances_to_flip, n_instances_to_reject):
        ordered_disc_scores = disc_scores.sort_values(ascending=False)

        if n_instances_to_flip > 0:
            cut_off_flip_score = ordered_disc_scores.iloc[n_instances_to_flip-1]
        else:
            cut_off_flip_score = np.inf

        #everything that's bigger than cut_off_disc_score -> reject
        if n_instances_to_reject > 0:
            cut_off_reject_score = ordered_disc_scores.iloc[n_instances_to_flip + n_instances_to_reject - 1]
        else:
            cut_off_reject_score = np.inf

        return cut_off_flip_score, cut_off_reject_score

    #if uncertainty > cut_off_reject_score -> reject
    def decide_on_reject_threshold_uncertain_data(self, uncertainty_scores, n_reject):
        ordered_uncertainty_scores = uncertainty_scores.sort_values(ascending=False)

        if n_reject > 0:
            cut_off_reject_score = ordered_uncertainty_scores.iloc[n_reject-1]
        else:
            cut_off_reject_score = np.inf

        return cut_off_reject_score



    def predict(self, test_dataset):
        #Step 1: Apply black box classifier, and store predictions
        test_data_with_preds = self.make_bb_preds_and_preds_proba_for_data(test_dataset)
        predictions = test_data_with_preds[self.decision_attribute]

        #Step 2: Check which instances fall under reject rules
        data_covered_by_rules, relevant_rules, relevant_slifts = self.extract_highest_slift_rule_per_instance(
            test_data_with_preds)

        #Step 3: Run situation testing on those instances
        sit_test_labels, sit_test_scores, sit_test_info = self.situationTester.predict_disc_labels(data_covered_by_rules)

        sit_test_scores = sit_test_scores.reindex(test_data_with_preds.index, fill_value=0)
        sit_test_info = sit_test_info.reindex(test_data_with_preds.index, fill_value = SituationTestingInfo(disc_score=-100))
        relevant_slifts = relevant_slifts.reindex(test_data_with_preds.index, fill_value=0)
        relevant_rules = relevant_rules.reindex(test_data_with_preds.index, fill_value=Rule(id=0, rule_base={}, rule_consequence={}))

        self.save_all_test_information_to_excel_file(test_data_with_preds, relevant_rules, sit_test_info)
        all_unfairness_based_rejected_indices = []
        all_unfairness_based_flipped_indices= []
        all_uncertainty_based_rejected_indices = []

        #Step 5: Apply the reject threshold
        for pd_itemset in self.intersectional_pd_itemsets:
            data_from_itemset = get_instances_covered_by_rule_base(pd_itemset.dict_notation, test_data_with_preds)
            neg_data_from_itemset = data_from_itemset[
                data_from_itemset[self.decision_attribute] == self.negative_label]
            pos_data_from_itemset = data_from_itemset[
                data_from_itemset[self.decision_attribute] == self.positive_label]

            print(pd_itemset)
            pos_ratio_for_itemset = len(pos_data_from_itemset )/len(data_from_itemset)
            print("N", len(data_from_itemset))
            print("N pos", len(pos_data_from_itemset))

            if pd_itemset in self.reference_group_list:
                unfair_part_of_data = pos_data_from_itemset
                fair_part_of_data = neg_data_from_itemset
            else:
                unfair_part_of_data = neg_data_from_itemset
                fair_part_of_data = pos_data_from_itemset


            slifts_for_itemset = relevant_slifts.loc[unfair_part_of_data.index]
            sit_test_scores_for_itemset = sit_test_scores.loc[unfair_part_of_data.index]
            uncertainty_scores_for_itemset = 1 - unfair_part_of_data['pred. probability']

            glu_dataset = pd.concat({'global_unf': slifts_for_itemset,'local_unf': sit_test_scores_for_itemset,'unc': uncertainty_scores_for_itemset}, axis=1)

            copula = GaussianMultivariate()
            copula.fit(glu_dataset)

            # compute joint CDF
            joint_cdf_score =  pd.Series(self.copula_per_group[pd_itemset].cumulative_distribution(glu_dataset) + np.random.uniform(-0.001, 0.001, size=len(unfair_part_of_data)), index=glu_dataset.index)

            flip_threshold = self.flip_thresholds_per_group[pd_itemset]
            reject_threshold = self.reject_threshold_per_group[pd_itemset]
            to_flip_unfair = joint_cdf_score[joint_cdf_score >= flip_threshold]
            to_reject_unfair = joint_cdf_score[(joint_cdf_score < flip_threshold) & (joint_cdf_score >= reject_threshold)]
            all_unfairness_based_flipped_indices.extend(to_flip_unfair.index)
            all_unfairness_based_rejected_indices.extend(to_reject_unfair.index)

            unc_neg_threshold = self.unc_neg_threshold_per_group[pd_itemset]
            unc_pos_threshold = self.unc_pos_threshold_per_group[pd_itemset]
            non_rejected_unfair_data_indices = joint_cdf_score[(joint_cdf_score < reject_threshold)].index
            non_rejected_unfair_data = unfair_part_of_data.loc[non_rejected_unfair_data_indices]
            all_non_rejected_data = pd.concat([non_rejected_unfair_data, fair_part_of_data])
            neg_preds_uncertainty = 1 - all_non_rejected_data[
                all_non_rejected_data[self.decision_attribute] == self.negative_label]['pred. probability']
            pos_preds_uncertainty = 1 - all_non_rejected_data[
                all_non_rejected_data[self.decision_attribute] == self.positive_label]['pred. probability']
            to_reject_unc_neg = neg_preds_uncertainty[neg_preds_uncertainty>unc_neg_threshold]
            to_reject_unc_pos = pos_preds_uncertainty[pos_preds_uncertainty>unc_pos_threshold]
            all_uncertainty_based_rejected_indices.extend(to_reject_unc_neg.index)
            all_uncertainty_based_rejected_indices.extend(to_reject_unc_pos.index)

        print("IFAC is rejecting: ", len(all_unfairness_based_rejected_indices) + len(all_uncertainty_based_rejected_indices), " instances")

        unf_rejected_instances = test_data_with_preds.loc[all_unfairness_based_rejected_indices]
        org_predictions_unf_rejected_instances = test_data_with_preds[self.decision_attribute].loc[all_unfairness_based_rejected_indices]
        org_predictions_probas_unf_rejected_instances = test_data_with_preds['pred. probability'].loc[all_unfairness_based_rejected_indices]
        sit_test_info_unf_rejected_instances = sit_test_info.loc[all_unfairness_based_rejected_indices]
        relevant_rules_unf_rejected_instances = relevant_rules.loc[all_unfairness_based_rejected_indices]

        unc_rejected_instances = test_data_with_preds.loc[all_uncertainty_based_rejected_indices]
        org_predictions_unc_rejected_instances = test_data_with_preds[self.decision_attribute].loc[
            all_uncertainty_based_rejected_indices]
        org_predictions_probas_unc_rejected_instances = test_data_with_preds['pred. probability'].loc[
            all_uncertainty_based_rejected_indices]

        unf_flipped_instances = test_data_with_preds.loc[all_unfairness_based_flipped_indices]
        org_predictions_to_be_flipped = test_data_with_preds[self.decision_attribute].loc[all_unfairness_based_flipped_indices]
        org_predictions_probas_unf_flipped_instances = test_data_with_preds['pred. probability'].loc[all_unfairness_based_flipped_indices]
        sit_test_info_unf_flipped_instances = sit_test_info.loc[all_unfairness_based_flipped_indices]
        relevant_rules_unf_flipped_instances = relevant_rules.loc[all_unfairness_based_flipped_indices]


        flipped_predictions = org_predictions_to_be_flipped.replace({self.negative_label: self.positive_label, self.positive_label: self.negative_label})

        all_unfairness_based_rejects_df = pd.DataFrame({
            'instance': [unf_rejected_instances.loc[i].to_dict() for i in all_unfairness_based_rejected_indices],
            'prediction_without_reject':  org_predictions_unf_rejected_instances,
            'prediction probability':  org_predictions_probas_unf_rejected_instances,
            'relevant_rule': relevant_rules_unf_rejected_instances,
            'sit_test_info': sit_test_info_unf_rejected_instances,
            }, index=all_unfairness_based_rejected_indices)

        all_unfairness_based_flips_df = pd.DataFrame({
            'instance': [unf_flipped_instances.loc[i].to_dict() for i in all_unfairness_based_flipped_indices],
            'prediction_without_reject':  org_predictions_to_be_flipped,
            'prediction probability':  org_predictions_probas_unf_flipped_instances,
            'relevant_rule': relevant_rules_unf_flipped_instances,
            'sit_test_info': sit_test_info_unf_flipped_instances,
            }, index=all_unfairness_based_flipped_indices)

        all_uncertainty_based_rejects_df = pd.DataFrame({
            'instance': [unc_rejected_instances.loc[i].to_dict() for i in all_uncertainty_based_rejected_indices],
            'prediction_without_reject': org_predictions_unc_rejected_instances,
            'prediction probability': org_predictions_probas_unc_rejected_instances,
        }, index=all_uncertainty_based_rejected_indices)

        all_unfairness_based_rejects_series = all_unfairness_based_rejects_df.apply(create_unfairness_based_reject, axis=1)
        all_uncertainty_based_rejects_series = (pd.Series([]) if all_uncertainty_based_rejects_df.empty
                                                else all_uncertainty_based_rejects_df.apply(
            create_uncertainty_based_reject, axis=1))
        all_unfairness_based_flips_series = all_unfairness_based_flips_df.apply(create_unfairness_based_flip, axis = 1)

        predictions.update(all_unfairness_based_rejects_series)
        predictions.update(all_uncertainty_based_rejects_series)
        predictions.update(flipped_predictions)

        return predictions, all_unfairness_based_rejects_series, all_uncertainty_based_rejects_series, all_unfairness_based_flips_series

    def calculate_flip_and_reject_rates_per_group(self, data_with_labels, data_with_preds):
        print("VALIDATION DATA")
        pos_decision_dict = {self.decision_attribute:self.positive_label}

        data_with_pos_decision = get_instances_covered_by_rule_base(pos_decision_dict, data_with_preds)
        target_pos_ratio = len(data_with_pos_decision)/len(data_with_preds)
        print("TARGET POS RATIO: " + str(target_pos_ratio))
        #
        # predictions_with_pos_decision = get_instances_covered_by_rule_base(pos_decision_dict, data_with_preds)
        # target_pos_ratio_bb = len(predictions_with_pos_decision) / len(data_with_preds)
        # print("TARGET POS RATIO ACCORDING TO PREDICTION: " + str(target_pos_ratio_bb))

        n_unfairness_reject_pd_itemset = {}
        n_unfairness_flip_pd_itemset = {}
        n_uncertain_neg_decision_reject_pd_itemset = {}
        n_uncertain_pos_decision_reject_pd_itemset = {}

        n_total_rejects = 0
        for itemset in self.intersectional_pd_itemsets:
            print(itemset)

            data_belonging_to_itemset = get_instances_covered_by_rule_base(itemset.dict_notation, data_with_preds)

            data_with_pos_decision = get_instances_covered_by_rule_base(pos_decision_dict, data_belonging_to_itemset)
            pos_ratio_for_itemset = len(data_with_pos_decision)/len(data_belonging_to_itemset)
            n_pos_for_itemset = len(data_with_pos_decision)
            n_itemset = len(data_belonging_to_itemset)
            print("N_pos: ", n_pos_for_itemset)
            print("N: ", n_itemset)

            max_reject = int((1-self.coverage) * n_itemset)

            if pos_ratio_for_itemset < target_pos_ratio:
                n_unfair_reject, n_unfair_flip, n_pos_uncertain_rej, n_neg_uncertain_rej = self.calc_n_rejection_for_deprived_group(target_pos_ratio, max_reject, n_itemset, n_pos_for_itemset, pos_ratio_for_itemset)

            else:
                n_unfair_reject, n_unfair_flip, n_pos_uncertain_rej, n_neg_uncertain_rej = self.calc_n_rejection_for_favoured_group(target_pos_ratio, max_reject, n_itemset, n_pos_for_itemset, pos_ratio_for_itemset)


            print("Unfair reject: ", n_unfair_reject)
            print("Unfair flip: ", n_unfair_flip)
            print("Uncertain neg reject: ", n_neg_uncertain_rej)
            print("Uncertain pos reject: ", n_pos_uncertain_rej)
            n_unfairness_reject_pd_itemset[itemset] = n_unfair_reject
            n_unfairness_flip_pd_itemset[itemset] = n_unfair_flip
            n_uncertain_neg_decision_reject_pd_itemset[itemset] = n_neg_uncertain_rej
            n_uncertain_pos_decision_reject_pd_itemset[itemset] = n_pos_uncertain_rej
            n_total_rejects += n_unfair_reject + n_neg_uncertain_rej + n_pos_uncertain_rej

        return n_unfairness_flip_pd_itemset, n_unfairness_reject_pd_itemset, n_uncertain_neg_decision_reject_pd_itemset, n_uncertain_pos_decision_reject_pd_itemset


    #calculating n_unfair_reject, n_unfair_flip, n_pos_uncertain_rej, n_neg_uncertain_rej for deprived group
    def calc_n_rejection_for_deprived_group(self, target_pos_ratio, max_reject, n_itemset, n_pos_for_itemset, pos_ratio_for_itemset):
        need_to_reject = int((n_pos_for_itemset - (target_pos_ratio * n_itemset)) / (-target_pos_ratio))
        if need_to_reject > max_reject:
            n_unfair_reject = max_reject
            n_unfair_flip = int((target_pos_ratio * n_itemset) - (target_pos_ratio * max_reject) - (
                    pos_ratio_for_itemset * n_itemset))
            n_pos_uncertain_rej = 0
            n_neg_uncertain_rej = 0
        else:
            n_unfair_reject = need_to_reject
            n_unfair_flip = 0
            leftover_reject = max_reject - n_unfair_reject
            n_pos_uncertain_rej = int(n_pos_for_itemset - target_pos_ratio * (n_itemset - max_reject))
            n_neg_uncertain_rej = leftover_reject - n_pos_uncertain_rej

        return n_unfair_reject, n_unfair_flip, n_pos_uncertain_rej, n_neg_uncertain_rej

    #calculating n_unfair_reject, n_unfair_flip, n_pos_uncertain_rej, n_neg_uncertain_rej
    def calc_n_rejection_for_favoured_group(self, target_pos_ratio, max_reject, n_itemset, n_pos_for_itemset, pos_ratio_for_itemset):
        need_to_reject = int((n_pos_for_itemset - (target_pos_ratio * n_itemset)) / (1 - target_pos_ratio))
        if need_to_reject > max_reject:
            n_unfair_reject = max_reject
            n_unfair_flip = int((pos_ratio_for_itemset * n_itemset) - (target_pos_ratio * n_itemset) + (
                        target_pos_ratio * max_reject) - max_reject)
            n_pos_uncertain_rej = 0
            n_neg_uncertain_rej = 0
        else:
            n_unfair_reject = need_to_reject
            n_unfair_flip = 0
            leftover_reject = max_reject - n_unfair_reject
            n_pos_uncertain_rej = int(
                n_pos_for_itemset - need_to_reject - target_pos_ratio * (n_itemset - max_reject))
            n_neg_uncertain_rej = leftover_reject - n_pos_uncertain_rej

        return n_unfair_reject, n_unfair_flip, n_pos_uncertain_rej, n_neg_uncertain_rej

    def check_if_decision_is_potentially_unfair(self, decision, pd_itemset):
        if decision == self.negative_label:
            return pd_itemset not in self.reference_group_list
        elif decision == self.positive_label:
            return pd_itemset in self.reference_group_list
        else:
            return False

    def save_all_test_information_to_excel_file(self, test_data, relevant_rules, relevant_sit_test_info):
        all_data = []
        for i in range(len(test_data)):
            not_rejected = False
            data_dict = test_data.loc[i].to_dict()

            decision_outcome = data_dict[self.decision_attribute]
            relevant_pd_itemset = return_relevant_pd_itemset(data_dict, self.sensitive_attributes)

            uncertainty_score = 1 - data_dict['pred. probability']
            relevant_rule = relevant_rules.loc[i]
            sit_test_info = relevant_sit_test_info.loc[i]

            data_dict["uncertainty_score"] = uncertainty_score
            GLU_score = uncertainty_score

            flip_threshold = self.flip_thresholds_per_group[relevant_pd_itemset]
            reject_threshold = self.reject_threshold_per_group[relevant_pd_itemset]
            unc_neg_threshold = self.unc_neg_threshold_per_group[relevant_pd_itemset]
            unc_pos_threshold = self.unc_pos_threshold_per_group[relevant_pd_itemset]

            is_potentially_unfair = self.check_if_decision_is_potentially_unfair(decision_outcome, relevant_pd_itemset)

            if is_potentially_unfair:
                data_dict['relevant_rule_id'] = relevant_rule.id
                data_dict['max_slift'] = relevant_rule.slift
                GLU_score += relevant_rule.slift

                data_dict['sit_test_score'] = sit_test_info.disc_score
                GLU_score += sit_test_info.disc_score
                data_dict['closest_favoured'] = sit_test_info.closest_reference
                data_dict['closest_discriminated'] = sit_test_info.closest_non_reference

                if GLU_score > flip_threshold:
                    data_dict["selector"] = "Fairness-Flip"
                elif GLU_score > reject_threshold:
                    data_dict["selector"] = "Fairness-Reject"
                else:
                    not_rejected = True

            else:
                data_dict['relevant_rule_id'] = -100
                data_dict['max_slift'] = 0
                data_dict['sit_test_score'] = 0
                data_dict['closest_favoured'] = []
                data_dict['closest_discriminated'] = []
                not_rejected = True


            if not_rejected:
                if decision_outcome == self.positive_label:
                    data_dict["selector"] = "Uncertainty-Reject" if uncertainty_score > unc_pos_threshold else "Keep"
                else:
                    data_dict["selector"] = "Uncertainty-Reject" if uncertainty_score > unc_neg_threshold else "Keep"

            data_dict["GLU_score"] = GLU_score
            all_data.append(data_dict)

            # Create a DataFrame
        df = pd.DataFrame(all_data)
        # Save to an Excel file
        df.to_excel(self.task+"test_info.xlsx", index=False, engine="openpyxl")


def save_reject_rules_in_excel_file(task, reject_rules):
    # Convert data_dict to a list of dictionaries
    data_rows = []
    for pd_itemset, rules in reject_rules.items():
        for rule in rules:
            rule_base_without_protected_itemset = deepcopy(rule.rule_base)
            for key in pd_itemset.dict_notation.keys():
                rule_base_without_protected_itemset.pop(key, None)

            data_rows.append({
                "id": str(rule.id),
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
    df.to_excel(task+"_reject_rules.xlsx", index=False, engine="openpyxl")



def return_relevant_pd_itemset(datapoint, sensitive_attributes):
    sens_attribute_dict = {}
    for sensitive_attribute in sensitive_attributes:
        sens_attribute_dict[sensitive_attribute] = datapoint[sensitive_attribute]

    return PD_itemset(sens_attribute_dict)