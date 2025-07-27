from load_datasets import load_income_data, load_german_credit, load_grade_prediction_data, load_OULAD, load_mortgage, load_recidivism, load_census_data
from IFAC.IFAC_Org import IFAC_Unfair_Uncertain_Rejects
from IFAC.IFAC_Alt import IFAC_Alt
from IFAC.IFAC_weighted import Weighted_IFAC
from IFAC.IFAC_Alt_GC import IFAC_Alt_GC
from IFAC.IFAC_Alt_withoutFlip import IFAC_Alt_NoFlip
from IFAC.Reject import Reject
from IFAC.BlackBoxClassifier import BlackBoxClassifier
from SchreuderSelectiveClassifier import SelectiveClassifierSchreuder
from IFAC.PD_itemset import generate_potentially_discriminated_itemsets, PD_itemset
from Baselines import UBAC, SCross, AUCPlugIn
from performance_measuring import extract_fairness_df_over_performance_dataframe, calculate_fairness_measures_over_averaged_performance_dataframe, extract_performance_df_over_non_rejected_instances, IFAC_reject_info_to_text_file, Schreuder_flip_info_to_text_file, average_performance_results_over_multiple_splits, average_fairness_results_over_multiple_splits
import pandas as pd
import numpy as np
from visualizations import visualize_averaged_performance_measure_for_single_and_intersectional_axis
import os
from copy import deepcopy



def save_for_gui(coverage, base_classifier):
    data = load_income_data()
    sensitive_attributes = ["sex", "race"]
    reference_group = {'sex': 'Male', 'race': 'White alone'}

    train_ratio = 0.7
    val_ratio = 0.25
    test_set_size = 1200
    number_of_test_sets = 2

    train_data, test_data_array = data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
                                                                               number_of_test_sets=number_of_test_sets,
                                                                               size_of_each_test_set=test_set_size)
    ifac = IFAC_Alt(coverage=coverage, sensitive_attributes=sensitive_attributes,
                           reference_group_list=[PD_itemset(reference_group)],
                           val1_ratio=val_ratio, val2_ratio=val_ratio, base_classifier=base_classifier)
    ifac.fit(train_data)


def save_for_gui(coverage, task, base_classifier):
    data_loading_function_dict = {"income": load_income_data, "census": load_census_data, "oulad": load_OULAD,
                                  "recidivism": load_recidivism, "mortgage": load_mortgage}
    sensitive_attributes_dict = {"income": ["sex", "race"], "census": ["sex"], "oulad": ["disability"],
                                 "recidivism": ["race"], "mortgage": ["derived_race"]}
    reference_group_dict = {"income": {'sex': 'Male', 'race': 'White alone'}, "census": {'sex': 'Male'},
                            "oulad": {'disability': 'N'}, "recidivism": {'race': 'Caucasian'},
                            "mortgage": {'derived_race': 'White'}}

    data = data_loading_function_dict[task]()
    sensitive_attributes = sensitive_attributes_dict[task]
    reference_group = reference_group_dict[task]

    train_ratio = 0.7
    val_ratio = 0.25
    test_set_size = 1200
    number_of_test_sets = 2

    train_data, test_data_array = data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
                                                                               number_of_test_sets=number_of_test_sets,
                                                                               size_of_each_test_set=test_set_size)

    pd_itemsets = generate_potentially_discriminated_itemsets(train_data, sensitive_attributes)
    pd_itemsets.append(PD_itemset({}))

    ifac = IFAC_Alt(coverage=coverage, sensitive_attributes=sensitive_attributes,
                           reference_group_list=[PD_itemset(reference_group)], task=task,
                           val1_ratio=val_ratio, val2_ratio=val_ratio, base_classifier=base_classifier)

    ifac.fit(train_data)
    for test_data in test_data_array:
        predictions, all_unfairness_based_rejects_series, all_uncertainty_based_rejects_series, all_unfairness_based_flips_series = ifac.predict(test_data)


def test_schreuder_stuff(coverage, base_classifier):
    data = load_OULAD()
    sensitive_attributes = ["disability"]
    reference_group = {'disability': 'N'}

    train_ratio = 0.7
    test_set_size = 1200
    number_of_test_sets = 2
    val_ratio = 0.25

    train_data, test_data_array = data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
                                                                               number_of_test_sets=number_of_test_sets,
                                                                               size_of_each_test_set=test_set_size)

    pd_itemsets = generate_potentially_discriminated_itemsets(train_data, sensitive_attributes)
    pd_itemsets.append(PD_itemset({}))

    schreuder = SelectiveClassifierSchreuder(sensitive_attributes=sensitive_attributes,
                                             reference_groups=[reference_group],
                                             reject_rate_per_sens_group={1: coverage, 0: coverage},
                                             base_classifier=base_classifier)

    schreuder.fit(train_data)

    ifac = IFAC_Alt(coverage=coverage, sensitive_attributes=sensitive_attributes,
                           reference_group_list=[PD_itemset(reference_group)],
                           val1_ratio=val_ratio, val2_ratio=val_ratio, base_classifier=base_classifier)
    ifac.fit(train_data)

    iteration = 1
    for test_data in test_data_array:
        schreuder.compare_and_predict(test_data)
        iteration_performances, iteration_fairness = run_all_baselines(test_data, pd_itemsets,
                                                                       {"Schreuder": schreuder, "IFAC": ifac},
                                                                       iteration=iteration, text_file=text_file)
        all_performances = pd.concat([all_performances, iteration_performances], ignore_index=True)
        all_fairness_measures = pd.concat([all_fairness_measures, iteration_fairness], ignore_index=True)
        iteration += 1


def run_baseline_classifier(task, base_classifier):
    data_loading_function_dict = {"income": load_income_data, "census": load_census_data, "oulad": load_OULAD,
                                  "recidivism": load_recidivism, "mortgage": load_mortgage}

    data = data_loading_function_dict[task]()
    sensitive_attributes = data.sensitive_attributes
    reference_group_list = [PD_itemset(data.reference_group_dict)]

    train_ratio = 0.6
    val_ratio = 0.25
    test_set_size = 0.5
    number_of_test_sets = 2

    train_data, sit_test, test_data_array = data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
                                                                                         number_of_test_sets=number_of_test_sets,
                                                                                         size_of_each_test_set=test_set_size)

    sit_test_data = sit_test.descriptive_data
    pd_itemsets = generate_potentially_discriminated_itemsets(train_data, sensitive_attributes)
    pd_itemsets.append(PD_itemset({}))
    BB = BlackBoxClassifier(base_classifier)
    BB.fit(train_data)

    all_performances = pd.DataFrame([])
    all_fairness_measures = pd.DataFrame([])
    iteration = 1

    for test_data in test_data_array:
        iteration_performances, iteration_fairness = run_all_baselines(test_data, sit_test_data, pd_itemsets, reference_group_list, {base_classifier: BB},
                                                                       iteration=iteration)
        all_performances = pd.concat([all_performances, iteration_performances], ignore_index=True)
        all_fairness_measures = pd.concat([all_fairness_measures, iteration_fairness], ignore_index=True)
        iteration += 1

    averaged_performances = average_performance_results_over_multiple_splits(all_performances)
    averaged_fairness_measures = average_fairness_results_over_multiple_splits(all_fairness_measures)

    relevant_performance_row = averaged_performances[
        (averaged_performances["Classification Type"] == base_classifier) & (averaged_performances["Group"] == "")]
    print("Accuracy:" + str(relevant_performance_row[["Classification Type", "Accuracy mean"]]))

    relevant_fairness_row = averaged_fairness_measures[
        (averaged_fairness_measures["Classification Type"] == base_classifier) & (
                averaged_fairness_measures["Sensitive Features"] == 'derived_race')]
    print("Group Unfairness:" + str(relevant_fairness_row[["Classification Type", "Highest Diff. in Pos. Ratio mean"]]))

    print("Individual Unfairness:" + str(relevant_performance_row[["Classification Type", "Avg. Situation Testing Non-Rejected mean"]]))



def run_experiment(task, coverage, base_classifier, name_test_run):
    data_loading_function_dict = {"income": load_income_data, "census": load_census_data, "oulad": load_OULAD, "recidivism": load_recidivism, "mortgage": load_mortgage}
    # sensitive_attributes_dict = {"income": ["sex", "race"], "census": ["sex"], "oulad": ["disability"], "recidivism": ["race"], "mortgage": ["derived_race"]}
    # reference_group_dict = {"income": {'sex': 'Male', 'race': 'White alone'}, "census": {'sex': 'Male'}, "oulad": {'disability': 'N'}, "recidivism": {'race': 'Caucasian'}, "mortgage": {'derived_race': 'White'}}

    data = data_loading_function_dict[task]()

    path = os.getcwd()
    parent_dir = os.path.dirname(path) + "\\IFAC\\final_results\\" + task
    name_test_run = "cov = " + str(coverage) + ", bb = " + str(base_classifier) + name_test_run
    path = os.path.join(parent_dir, name_test_run)
    os.mkdir(path)
    text_file = open(path + "\\info.txt", 'w')

    train_ratio = 0.6
    val_ratio = 0.25
    test_set_size = 0.5
    number_of_test_sets = 20 #20

    train_data, sit_test, test_data_array = data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
                                                                               number_of_test_sets=number_of_test_sets,
                                                                               size_of_each_test_set=test_set_size)

    sit_test_data = sit_test.descriptive_data

    pd_itemsets = generate_potentially_discriminated_itemsets(train_data, data.sensitive_attributes)
    pd_itemsets.append(PD_itemset({}))

    ubac = UBAC(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
    ubac.fit(train_data)

    schreuder = SelectiveClassifierSchreuder(reject_rate_per_sens_group={1: coverage, 0: coverage},
                                             base_classifier=base_classifier)
    schreuder.fit(train_data)

    scross = SCross(base_classifier, coverage=coverage)
    scross.fit(train_data)

    auc = AUCPlugIn(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
    auc.fit(train_data)

    ifac = Weighted_IFAC(coverage=coverage, task=task, val1_ratio=val_ratio, val2_ratio=val_ratio, base_classifier=base_classifier)
    ifac.fit_discriminatory_associations(train_data)

    equal_weights_ifac = deepcopy(ifac)
    equal_weights_ifac.fit_reject_thresholds(w1=0.33, w2=0.33, w3=0.33)

    only_fairness_ifac = deepcopy(ifac)
    only_fairness_ifac.fit_reject_thresholds(w1=0.5, w2=0.5, w3=0.0)

    only_uncertainty_ifac = deepcopy(ifac)
    only_uncertainty_ifac.fit_reject_thresholds(w1=0.0, w2=0.0, w3=1.0)


    all_performances = pd.DataFrame([])
    all_fairness_measures = pd.DataFrame([])
    iteration = 1
    for test_data in test_data_array:
        print("Iteration: ", iteration)
        iteration_performances, iteration_fairness = run_all_baselines(test_data, sit_test_data, pd_itemsets,  reference_group_list=[PD_itemset(data.reference_group_dict)],
                                                                       name_sel_classifier_dict={"PlugIn": ubac,  "AUC": auc, "SCross": scross, "Schreuder": schreuder, "IFAC-GLU": equal_weights_ifac, "IFAC_GL": only_fairness_ifac, "IFAC_U": only_uncertainty_ifac},   #"PlugIn": ubac,  "AUC": auc, "SCross": scross, "Schreuder": schreuder, "IFAC": ifac, "IFAC-NoFlip": ifac_no_flip
                                                                       iteration=iteration, text_file=text_file)
        all_performances = pd.concat([all_performances, iteration_performances], ignore_index=True)
        all_fairness_measures = pd.concat([all_fairness_measures, iteration_fairness], ignore_index=True)
        iteration += 1

    averaged_performances = average_performance_results_over_multiple_splits(all_performances)
    averaged_fairness_measures = average_fairness_results_over_multiple_splits(all_fairness_measures)

    averaged_performances.to_excel(path + "\\performances.xlsx")
    averaged_fairness_measures.to_excel(path+"\\fairness.xlsx")

    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,
                                                                              "Positive Dec. Ratio",
                                                                              path_to_save_figure=path)
    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR",
                                                                              path_to_save_figure=path)
    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR",
                                                                              path_to_save_figure=path)

    return

def run_all_baselines(test_data, sit_test_data, pd_itemsets,  reference_group_list, name_sel_classifier_dict, text_file=None, iteration=0):
    all_performances = pd.DataFrame([])
    for name, classifier in name_sel_classifier_dict.items():
        if name.startswith("IFAC"):
            ifac_predictions, unf_based_rejects_info, unc_based_rejects_info, unf_based_flips_info = classifier.predict(test_data)
            classifier_performance = extract_performance_df_over_non_rejected_instances(classification_method=name,
                                                                                        data=test_data, sit_test_data=sit_test_data,
                                                                                        predictions=ifac_predictions,
                                                                                        pd_itemsets=pd_itemsets, reference_group_list=reference_group_list)
            IFAC_reject_info_to_text_file(iteration, unc_based_rejects_info, unf_based_rejects_info,
                                          unf_based_flips_info, text_file)
        elif name == "Schreuder":
            schreuder_predictions, schreuder_flips, schreuder_rejects = classifier.predict(test_data)
            classifier_performance = extract_performance_df_over_non_rejected_instances(classification_method=name, data=test_data,
                                                                              sit_test_data=sit_test_data, predictions=schreuder_predictions,
                                                                              pd_itemsets=pd_itemsets, reference_group_list=reference_group_list)
            Schreuder_flip_info_to_text_file(iteration, schreuder_flips, schreuder_rejects, text_file)
        else:
            classifier_preds = classifier.predict(test_data)
            classifier_performance = extract_performance_df_over_non_rejected_instances(classification_method=name, data=test_data,
                                                                              sit_test_data=sit_test_data, predictions=classifier_preds,
                                                                              pd_itemsets=pd_itemsets, reference_group_list=reference_group_list)
        all_performances = pd.concat([all_performances, classifier_performance], ignore_index=True)
    fairness_dataframe = extract_fairness_df_over_performance_dataframe(all_performances)
    return all_performances, fairness_dataframe


