from IFAC.Reject import Reject
from IFAC.PD_itemset import PD_itemset
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from IFAC.Rule import get_instances_covered_by_rule_base
from IFAC.SituationTesting import SituationTesting
from copy import deepcopy

def extract_fairness_df_over_performance_dataframe(performance_dataframe):
    fairness_performance_dataframe = pd.DataFrame([], columns = ["Classification Type", "Sensitive Features", "Highest Diff. in Pos. Ratio", "Highest Diff. in FPR", "Highest Diff. in FNR"])
    dataframes_split_by_sens_features = dict(tuple(performance_dataframe.groupby("Sensitive Features")))
    for sensitive_feature_key, dataframe in dataframes_split_by_sens_features.items():
        if sensitive_feature_key == "":
            continue
        dataframes_split_by_classification_type = dict(tuple(dataframe.groupby('Classification Type')))
        for classification_type_key, dataframe in dataframes_split_by_classification_type.items():
            highest_fpr = dataframe['FPR'].max()
            lowest_fpr = dataframe['FPR'].min()
            highest_diff_in_fpr = highest_fpr - lowest_fpr
            standard_deviation_fpr = dataframe['FPR'].std()

            highest_fnr = dataframe['FNR'].max()
            lowest_fnr = dataframe['FNR'].min()
            highest_diff_in_fnr = highest_fnr - lowest_fnr
            standard_deviation_fnr = dataframe['FNR'].std()

            highest_pos_dec = dataframe['Positive Dec. Ratio'].max()
            lowest_pos_dec = dataframe['Positive Dec. Ratio'].min()
            highest_diff_in_pos_dec = highest_pos_dec - lowest_pos_dec
            standard_deviation_pos_decision_ratio = dataframe['Positive Dec. Ratio'].std()

            row_entry = {"Classification Type": classification_type_key, "Sensitive Features" : sensitive_feature_key,
                         "Highest Diff. in Pos. Ratio": highest_diff_in_pos_dec,
                         "Std. Pos. Ratio" : standard_deviation_pos_decision_ratio,
                         "Highest Diff. in FPR" : highest_diff_in_fpr,
                         "Std. FPR": standard_deviation_fpr,
                         "Highest Diff. in FNR": highest_diff_in_fnr,
                         "Std. FNR": standard_deviation_fnr}
            row_df = pd.DataFrame([row_entry])

            fairness_performance_dataframe = pd.concat([fairness_performance_dataframe, row_df], ignore_index=True)
    return fairness_performance_dataframe

def select_potentially_unfair_decisions(relevant_data, decision_attribute, desirable_label, reference_group_list):
    relevant_data = deepcopy(relevant_data)
    all_reference_group_data = pd.DataFrame([])
    for reference_group in reference_group_list:
        reference_group_data = get_instances_covered_by_rule_base(reference_group.dict_notation, relevant_data)
        all_reference_group_data = pd.concat([all_reference_group_data, reference_group_data], axis=0)
        relevant_data = relevant_data.drop(reference_group_data.index)
    non_reference_group_data = relevant_data

    positive_decisions_from_reference_group = all_reference_group_data[all_reference_group_data[decision_attribute] == desirable_label]
    negative_decisions_from_non_reference_group = non_reference_group_data[non_reference_group_data[decision_attribute] != desirable_label]

    potentially_unfair_data = pd.concat([positive_decisions_from_reference_group, negative_decisions_from_non_reference_group])
    return potentially_unfair_data


def extract_performance_df_over_non_rejected_instances(classification_method, data, sit_test_data, predictions, pd_itemsets, reference_group_list):
    desirable_label = data.desirable_label
    undesirable_label = data.undesirable_label
    decision_label = data.decision_attribute
    ground_truth = data.descriptive_data[data.decision_attribute]

    rejected_predictions = predictions[predictions.apply(lambda x: isinstance(x, Reject))]
    rejected_part_of_data = data.descriptive_data.loc[rejected_predictions.index]
    original_predictions_before_rejects = rejected_predictions.apply(lambda x: x.prediction_without_reject)

    non_rejected_predictions = predictions[predictions.apply(lambda x: not isinstance(x, Reject))]
    non_rejected_part_of_data = data.descriptive_data.loc[non_rejected_predictions.index]

    non_rejected_data_with_preds = deepcopy(non_rejected_part_of_data)
    non_rejected_data_with_preds = non_rejected_data_with_preds.drop(columns=[decision_label])
    non_rejected_data_with_preds[decision_label] = non_rejected_predictions

    rejected_data_with_org_preds = deepcopy(rejected_part_of_data)
    rejected_data_with_org_preds = rejected_data_with_org_preds.drop(columns=[decision_label])
    rejected_data_with_org_preds[decision_label] = original_predictions_before_rejects

    situation_testing = SituationTesting(reference_group_list=reference_group_list, decision_label=decision_label,desirable_label=desirable_label, k=10, t=0, distance_function=data.distance_function)
    #first need to make sure that non_rejected_part_of_data is given right
    situation_testing.fit(sit_test_data)

    potentially_unfair_non_rejected_data = select_potentially_unfair_decisions(non_rejected_data_with_preds, decision_label, desirable_label, reference_group_list)
    potentially_unfair_rejected_data = select_potentially_unfair_decisions(rejected_data_with_org_preds, decision_label, desirable_label, reference_group_list)

    _, disc_scores_non_rejected, _ = situation_testing.predict_disc_labels(potentially_unfair_non_rejected_data)
    disc_scores_non_rejected = disc_scores_non_rejected.reindex(non_rejected_part_of_data.index, fill_value=0)
    _, disc_scores_rejected, _ = situation_testing.predict_disc_labels(potentially_unfair_rejected_data)
    disc_scores_rejected = disc_scores_rejected.reindex(rejected_part_of_data.index, fill_value = 0)

    performance_df = pd.DataFrame([])

    for protected_itemset in pd_itemsets:
        performance_entry = {"Classification Type": classification_method, "Group": protected_itemset.string_notation,
                             "Sensitive Features": protected_itemset.sensitive_features}
        predicted_data_protected_itemset = get_instances_covered_by_rule_base(protected_itemset.dict_notation, non_rejected_part_of_data)
        predicted_indices_of_protected_itemset = predicted_data_protected_itemset.index
        rejected_indices_of_protected_itemset = get_instances_covered_by_rule_base(protected_itemset.dict_notation, rejected_part_of_data).index

        predictions_for_protected_itemset = predictions[predicted_indices_of_protected_itemset]
        ground_truth_for_protected_itemset = ground_truth[predicted_indices_of_protected_itemset]

        disc_scores_for_non_rejected_protected_itemset = disc_scores_non_rejected[predicted_indices_of_protected_itemset]
        disc_scores_for_non_rejected_protected_itemset = disc_scores_for_non_rejected_protected_itemset[disc_scores_for_non_rejected_protected_itemset >= 0]

        disc_scores_for_rejected_protected_itemset = disc_scores_rejected[rejected_indices_of_protected_itemset]
        disc_scores_for_rejected_protected_itemset = disc_scores_for_rejected_protected_itemset[disc_scores_for_rejected_protected_itemset >= 0]

        conf_matrix = confusion_matrix(ground_truth_for_protected_itemset, predictions_for_protected_itemset,
                                       labels=[desirable_label, undesirable_label])

        performance_entry["Coverage"] = len(predicted_indices_of_protected_itemset) / (len(rejected_indices_of_protected_itemset) + len(predicted_indices_of_protected_itemset))
        performance_entry["Accuracy"] = calculate_accuracy_based_on_conf_matrix(conf_matrix)
        performance_entry["Positive Dec. Ratio"] = calculate_positive_decision_ratio_based_on_conf_matrix(conf_matrix)
        performance_entry["FNR"] = calculate_false_negative_rate_based_on_conf_matrix(conf_matrix)
        performance_entry["FPR"] = calculate_false_positive_rate_based_on_conf_matrix(conf_matrix)
        performance_entry["Number of instances"] = calculate_number_of_instances_based_on_conf_matrix(conf_matrix)

        performance_entry["Avg. Situation Testing Non-Rejected"] = sum((disc_scores_for_non_rejected_protected_itemset)) / len(disc_scores_for_non_rejected_protected_itemset)

        if len(disc_scores_for_rejected_protected_itemset != 0):
            performance_entry["Avg. Situation Testing Rejected"] = sum((disc_scores_for_rejected_protected_itemset)) / len(disc_scores_for_rejected_protected_itemset)
        else:
            performance_entry["Avg. Situation Testing Rejected"] = -1
        print(performance_entry)

        # if protected_itemset == PD_itemset({}):
        #     performance_entry["Avg. Situation Testing"] = average_disc_scores
        #     print(performance_entry)
        # else:
        #     performance_entry["Avg. Situation Testing"] = -1

        performance_df = pd.concat([performance_df, pd.DataFrame([performance_entry])], ignore_index=True)


    return performance_df




def make_confusion_matrix_for_every_protected_itemset(desirable_label, undesirable_label, ground_truth, predicted_labels, protected_info, protected_itemsets, print_matrix=False):
    conf_matrix_dict = {}

    for protected_itemset in protected_itemsets:
        protected_itemset_dict = protected_itemset.dict_notation
        indices_belonging_to_this_pi = get_instances_covered_by_rule_base(protected_itemset_dict, protected_info).index
        ground_truth_of_indices = ground_truth.loc[indices_belonging_to_this_pi]
        predictions_for_indices = predicted_labels.loc[indices_belonging_to_this_pi]
        conf_matrix = confusion_matrix(ground_truth_of_indices, predictions_for_indices, labels=[desirable_label, undesirable_label])
        conf_matrix_dict[protected_itemset] = conf_matrix
        if print_matrix:
            print(protected_itemset)
            print(f"Total number of instances: {calculate_number_of_instances_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"Positive Decision Ratio: {calculate_positive_decision_ratio_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"False Positive Rate: {calculate_false_positive_rate_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"False Negative Rate: {calculate_false_negative_rate_based_on_conf_matrix(conf_matrix):.2f}")
            print(conf_matrix)
            print("_________")
    return conf_matrix_dict


def calculate_accuracy_based_on_conf_matrix(conf_matrix):
    number_true_negatives = conf_matrix[1][1]
    number_true_positives = conf_matrix[0][0]

    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]

    accuracy = (number_true_negatives + number_true_positives) / total
    return accuracy


def calculate_positive_decision_ratio_based_on_conf_matrix(conf_matrix):
    number_false_positives = conf_matrix[1][0]
    number_true_positives = conf_matrix[0][0]

    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]

    pos_ratio = (number_false_positives + number_true_positives) / total
    return pos_ratio


def calculate_false_positive_rate_based_on_conf_matrix(conf_matrix):
    number_false_positives = conf_matrix[1][0]
    number_true_negatives = conf_matrix[1][1]

    fpr = (number_false_positives) / (number_false_positives + number_true_negatives)
    return fpr


def calculate_false_negative_rate_based_on_conf_matrix(conf_matrix):
    number_false_negatives = conf_matrix[0][1]
    number_true_positives = conf_matrix[0][0]

    fnr = (number_false_negatives) / (number_false_negatives + number_true_positives)
    return fnr

def calculate_recall_based_on_conf_matrix(conf_matrix):
    number_true_positives = conf_matrix[0][0]
    number_false_negatives = conf_matrix[0][1]

    number_of_actual_positives = number_true_positives + number_false_negatives
    recall = number_true_positives/number_of_actual_positives
    return recall

def calculate_precision_based_on_conf_matrix(conf_matrix):
    number_true_positives = conf_matrix[0][0]
    number_false_positives = conf_matrix[1][0]

    number_of_predicted_positives = number_true_positives + number_false_positives
    precision = number_true_positives/number_of_predicted_positives
    return precision#

def calculate_number_of_instances_based_on_conf_matrix(conf_matrix):
    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]
    return total



def average_fairness_results_over_multiple_splits(fairness_dataframes):
    fairness_measures_of_interest = ["Highest Diff. in Pos. Ratio", "Highest Diff. in FPR", "Highest Diff. in FNR", "Std. Pos. Ratio", "Std. FPR", "Std. FNR"]
    summary_df = fairness_dataframes.groupby(['Classification Type', 'Sensitive Features']).agg(
        {"Highest Diff. in Pos. Ratio": ['mean', 'std'], "Highest Diff. in FPR": ['mean', 'std'],
         "Highest Diff. in FNR": ['mean', 'std'],
         "Std. Pos. Ratio": ['mean', 'std'], "Std. FNR": ['mean', 'std'], "Std. FPR": ['mean', 'std']}).reset_index()

    # summary_df = fairness_dataframes.groupby([['Classification Type', "Sensitive Features"]])[
    #     "Sensitive Features", "Highest Diff. in Pos. Ratio", "Highest Diff. in FPR", "Highest Diff. in FNR",
    #     "Std. FPR", "Std. FNR", "Std. Pos. Ratio"].agg(
    #     {"Highest Diff. in Pos. Ratio": ['mean', 'std'], "Highest Diff. in FPR": ['mean', 'std'],
    #      "Highest Diff. in FNR": ['mean', 'std'],
    #      "Std. Pos. Ratio": ['mean', 'std'], "Std. FNR": ['mean', 'std'], "Std. FPR": ['mean', 'std']})
    summary_df.columns = [' '.join(col).strip() for col in summary_df.columns.values]
    summary_df.reset_index(inplace=True)

    # calculate upper and lower bounds of confidence intervals
    for fairness_measure in fairness_measures_of_interest:
        summary_df[fairness_measure + ' ci'] = 1.96 * (
                summary_df[fairness_measure + ' std'] / np.sqrt(len(fairness_dataframes)))
    return summary_df


def average_performance_results_over_multiple_splits(performance_dataframes):
    performance_measures_of_interest = ['Coverage', 'Accuracy', 'Avg. Situation Testing Non-Rejected', 'Avg. Situation Testing Rejected', 'FPR', 'FNR', 'Positive Dec. Ratio', 'Number of instances']
    # summary_df = performance_dataframes.groupby(['Classification Type', 'Group', "Sensitive Features"])[
    #     "Group", "Sensitive Features", "Coverage", "Accuracy", "Avg. Situation Testing", "FPR", "FNR", "Positive Dec. Ratio", "Number of instances"].agg(
    #     {'Coverage': ['mean', 'std'], 'Accuracy': ['mean', 'std'], "Avg. Situation Testing": ['mean', 'std'], 'FPR': ['mean', 'std'], 'FNR': ['mean', 'std'], 'Positive Dec. Ratio': ['mean', 'std'],
    #      'Number of instances': ['mean', 'std']})
    summary_df = performance_dataframes.groupby(['Classification Type', 'Group', 'Sensitive Features']).agg({
        'Coverage': ['mean', 'std'],
        'Accuracy': ['mean', 'std'],
        'Avg. Situation Testing Non-Rejected': ['mean', 'std'],
        'Avg. Situation Testing Rejected': ['mean', 'std'],
        'FPR': ['mean', 'std'],
        'FNR': ['mean', 'std'],
        'Positive Dec. Ratio': ['mean', 'std'],
        'Number of instances': ['mean', 'std']
    }).reset_index()
    summary_df.columns = [' '.join(col).strip() for col in summary_df.columns.values]
    summary_df.reset_index(inplace=True)

    # calculate upper and lower bounds of confidence intervals
    for performance_measure in performance_measures_of_interest:
        summary_df[performance_measure + ' ci'] = 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(summary_df)))
        summary_df[performance_measure + ' ci_low'] = summary_df[performance_measure + ' mean'] - 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(summary_df)))
        summary_df[performance_measure + ' ci_high'] = summary_df[performance_measure + ' mean'] + 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(summary_df)))

        if performance_measure != "Number of instances":
            # make sure confidence intervals range from 0 to 1
            summary_df[performance_measure + ' ci_low'] = summary_df[
                performance_measure + ' ci_low'].apply(lambda x: 0 if x < 0 else x)
            summary_df[performance_measure + ' ci_high'] = summary_df[
                performance_measure + ' ci_high'].apply(lambda x: 1 if x > 1 else x)

    return summary_df


def average_diff_to_best_performance(best_performance, other_performances):
    sum_performance_differences = 0
    for other_performance in other_performances:
        sum_performance_differences += abs(best_performance - other_performance)

    avg_performance_difference = sum_performance_differences / (len(other_performances))
    return avg_performance_difference

def calculate_fairness_measures_over_averaged_performance_dataframe(performance_dataframe):
    fairness_performance_dataframe = pd.DataFrame([], columns = ["Classification Type", "Sensitive Features", "Highest Diff. in Pos. Ratio", "Highest Diff. in FPR", "Highest Diff. in FNR"])
    dataframes_split_by_sens_features = dict(tuple(performance_dataframe.groupby("Sensitive Features")))
    for sensitive_feature_key, dataframe in dataframes_split_by_sens_features.items():
        dataframes_split_by_classification_type = dict(tuple(dataframe.groupby('Classification Type')))
        for classification_type_key, dataframe in dataframes_split_by_classification_type.items():
            highest_fpr = dataframe['FPR mean'].max()
            lowest_fpr = dataframe['FPR mean'].min()
            average_diff_to_lowest_fpr = average_diff_to_best_performance(lowest_fpr, dataframe['FPR mean'])
            highest_diff_in_fpr = highest_fpr - lowest_fpr
            standard_deviation_fpr = dataframe['FPR mean'].std()

            highest_fnr = dataframe['FNR mean'].max()
            lowest_fnr = dataframe['FNR mean'].min()
            average_diff_to_lowest_fnr = average_diff_to_best_performance(lowest_fnr, dataframe['FNR mean'])
            highest_diff_in_fnr = highest_fnr - lowest_fnr
            standard_deviation_fnr = dataframe['FNR mean'].std()

            highest_pos_dec = dataframe['Positive Dec. Ratio mean'].max()
            lowest_pos_dec = dataframe['Positive Dec. Ratio mean'].min()
            average_diff_to_highest_pos_dec = average_diff_to_best_performance(highest_pos_dec, dataframe['Positive Dec. Ratio mean'])
            highest_diff_in_pos_dec = highest_pos_dec - lowest_pos_dec
            standard_deviation_pos_decision_ratio = dataframe['Positive Dec. Ratio mean'].std()

            row_entry = {"Classification Type": classification_type_key, "Sensitive Features" : sensitive_feature_key,
                         "Highest Diff. in Pos. Ratio": highest_diff_in_pos_dec, "Average Diff. to Highest Pos. Ratio": average_diff_to_highest_pos_dec,
                         "Std. Pos. Ratio" : standard_deviation_pos_decision_ratio,
                         "Highest Diff. in FPR" : highest_diff_in_fpr, "Average Diff. to Lowest FPR" : average_diff_to_lowest_fpr,
                         "Std. FPR": standard_deviation_fpr,
                         "Highest Diff. in FNR": highest_diff_in_fnr,  "Average Diff. to Lowest FNR" : average_diff_to_lowest_fnr,
                         "Std. FNR": standard_deviation_fnr}

            fairness_performance_dataframe = pd.concat([fairness_performance_dataframe, pd.DataFrame([row_entry])], ignore_index=True)
            #fairness_performance_dataframe = fairness_performance_dataframe.append(row_entry, ignore_index=True)
    #make sure things are in the right order
    fairness_performance_dataframe = fairness_performance_dataframe[["Classification Type", "Sensitive Features", "Highest Diff. in Pos. Ratio",
                                                                     "Average Diff. to Highest Pos. Ratio", "Std. Pos. Ratio", "Highest Diff. in FPR",
                                                                     "Average Diff. to Lowest FPR", "Std. FPR", "Highest Diff. in FNR",
                                                                     "Average Diff. to Lowest FNR", "Std. FNR"]]
    return fairness_performance_dataframe


def IFAC_reject_info_to_text_file(iteration, uncertainty_reject_info, unfairness_reject_info, unfairness_flips_info, text_file):
    text_file.write("Information on IFAC iteration " + str(iteration) + ":")

    text_file.write("---------------Uncertainty Rejects---------------")

    for uncertainty_reject in uncertainty_reject_info:
        text_file.write(str(uncertainty_reject))
        text_file.write("\n")

    text_file.write("---------------Unfairness Rejects--------------- \n")
    for unfairness_reject in unfairness_reject_info:
        text_file.write(str(unfairness_reject))
        text_file.write("\n")

    text_file.write("---------------Unfairness Flips--------------- \n")
    for unfairness_flip in unfairness_flips_info:
        text_file.write(str(unfairness_flip))
        text_file.write("\n")


def Schreuder_flip_info_to_text_file(iteration, flips_info, rejects_info, text_file):
    text_file.write("Information on Schreuder iteration " + str(iteration) + ":")

    text_file.write("---------------Flips--------------- \n")

    for flip in flips_info:
        text_file.write(str(flip))
        text_file.write("\n")

    text_file.write("---------------Rejects--------------- \n")

    for reject in rejects_info:
        text_file.write(str(reject))
        text_file.write("\n")



