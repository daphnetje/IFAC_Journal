# def compare_income_prediction(coverage, base_classifier, name_test_run):
#     income_prediction_data = load_income_data()
#     path = os.getcwd()
#     parent_dir = os.path.dirname(path) + r"\IFAC\results\income_prediction"
#     name_test_run = "cov = " + str(coverage) + ", bb = " + str(base_classifier) + name_test_run
#     path = os.path.join(parent_dir, name_test_run)
#     os.mkdir(path)
#     text_file = open(path + "\\info.txt", 'w')
#
#     train_ratio = 0.7
#     val_ratio = 0.25
#     test_set_size = 0.5
#     # test_set_size = int(len(income_prediction_data.descriptive_data) * train_ratio * val_ratio)
#     # print(test_set_size)
#     train_data, test_data_array = income_prediction_data.split_into_train_and_multiple_test_sets(train_size=train_ratio, number_of_test_sets=20, size_of_each_test_set=test_set_size)
#
#     pd_itemsets = generate_potentially_discriminated_itemsets(train_data, ['sex', 'race'])
#     pd_itemsets.append(PD_itemset({}))
#
#     ubac = UBAC(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
#     ubac.fit(train_data)
#     #
#     schreuder = SelectiveClassifierSchreuder(sensitive_attributes=['sex', 'race'], reference_groups=[{'sex': 'Male', 'race': 'White alone'}], reject_rate_per_sens_group={1: coverage, 0:coverage}, base_classifier=base_classifier)
#     schreuder.fit(train_data)
#     #
#     scross = SCross(base_classifier, coverage=coverage)
#     scross.fit(train_data)
#
#     auc = AUCPlugIn(coverage=coverage, val_ratio = val_ratio, base_classifier=base_classifier)
#     auc.fit(train_data)
#
#     ifac = IFAC_Alt_NoFlip(coverage=coverage, sensitive_attributes=['sex', 'race'],
#                      reference_group_list=[PD_itemset({'sex': 'Male', 'race': 'White alone'})],
#                      val1_ratio=val_ratio, val2_ratio=val_ratio, base_classifier=base_classifier)
#     ifac.fit(train_data)
#
#     all_performances = pd.DataFrame([])
#     iteration = 1
#     for test_data in test_data_array:
#         print("Iteration: ", iteration)
#         iteration_performances = run_all_baselines(test_data, pd_itemsets, {"PlugIn": ubac, "Schreuder": schreuder, "SCross": scross, "AUC": auc, "IFAC": ifac}, text_file=text_file, iteration=iteration)
#         all_performances = pd.concat([all_performances] + iteration_performances, ignore_index=True)
#         iteration += 1
#     averaged_performances = average_performance_results_over_multiple_splits(all_performances)
#     fairness_measures_over_average_performances = calculate_fairness_measures_over_averaged_performance_dataframe(
#         averaged_performances)
#
#
#     averaged_performances.to_excel(path + "\\performances.xlsx")
#     fairness_measures_over_average_performances.to_excel(path + "\\fairness.xlsx")
#
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,"Positive Dec. Ratio", path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR", path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR", path_to_save_figure=path)
#
#
# def compare_OULAD(coverage, base_classifier, name_test_run=""):
#     oulad_data = load_OULAD()
#     path = os.getcwd()
#     parent_dir = os.path.dirname(path) + r"\IFAC\results\oulad_prediction"
#     name_test_run = "cov = " + str(coverage) + ", bb = " + str(base_classifier) + name_test_run
#     path = os.path.join(parent_dir, name_test_run)
#     os.mkdir(path)
#     text_file = open(path + "\\info.txt" , 'w')
#
#     train_ratio = 0.7
#     val_ratio = 0.25
#     test_set_size = 0.5
#
#     train_data, test_data_array = oulad_data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
#                                                                                     number_of_test_sets=20, size_of_each_test_set=test_set_size)
#     pd_itemsets = generate_potentially_discriminated_itemsets(train_data, ['disability'])
#     pd_itemsets.append(PD_itemset({}))
#
#     ubac = UBAC(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
#     ubac.fit(train_data)
#
#     auc = AUCPlugIn(coverage=coverage, val_ratio = val_ratio, base_classifier=base_classifier)
#     auc.fit(train_data)
#
#     # scross = SCross(coverage=coverage, base_classifier=base_classifier, )
#     # scross.fit(train_data)
#     # #
#     # schreuder = SelectiveClassifierSchreuder(sensitive_attributes=["disability"], reference_groups=[{'disability': 'N'}], reject_rate_per_sens_group={1: coverage, 0:coverage}, base_classifier=base_classifier)
#     # schreuder.fit(train_data)
#     # #
#     # ifac = IFAC_Alt_NoFlip(coverage=coverage, sensitive_attributes=['disability'], reference_group_list=[PD_itemset({'disability': 'N'})], val1_ratio=0.25, val2_ratio=0.25, base_classifier=base_classifier)
#     # ifac.fit(train_data)
#
#     all_performances = pd.DataFrame([])
#     all_fairness_measures = pd.DataFrame([])
#     iteration = 1
#     for test_data in test_data_array:
#         print("Iteration: " + str(iteration)) #, "SCross": scross, "Schreuder": schreuder, "IFAC": ifac
#         iteration_performances, iteration_fairness = run_all_baselines(test_data, pd_itemsets, {"PlugIn": ubac, "AUC": auc}, iteration = iteration, text_file=text_file)
#         all_performances = all_performances.append(iteration_performances, ignore_index=True)
#         all_fairness_measures = all_fairness_measures.append(iteration_fairness, ignore_index=True)
#         #all_performances = pd.concat([all_performances] + iteration_performances, ignore_index=True)
#         iteration += 1
#
#     averaged_performances = average_performance_results_over_multiple_splits(all_performances)
#     average_fairnesses = average_fairness_results_over_multiple_splits(all_fairness_measures)
#     fairness_measures_over_average_performances = calculate_fairness_measures_over_averaged_performance_dataframe(averaged_performances)
#
#     averaged_performances.to_excel(path + "\\performances.xlsx")
#     average_fairnesses.to_excel(path + "\\fairness.xlsx")
#     fairness_measures_over_average_performances.to_excel(path + "\\OLDfairness.xlsx")
#
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,"Positive Dec. Ratio", path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR", path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR", path_to_save_figure=path)
#
#
#
#
# def compare_mortgage(coverage, base_classifier, name_test_run=""):
#     mortgage_data = load_mortgage()
#     path = os.getcwd()
#     parent_dir = os.path.dirname(path) + r"\IFAC\mortgage_prediction"
#     name_test_run = "cov = " + str(coverage) + ", bb = " + str(base_classifier) + name_test_run
#     path = os.path.join(parent_dir, name_test_run)
#     os.mkdir(path)
#     text_file = open(path + "\\info.txt" , 'w')
#
#     train_ratio = 0.7
#     val_ratio = 0.25
#     test_set_size = 0.5
#
#     train_data, test_data_array = mortgage_data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
#                                                                                         number_of_test_sets=5)
#     pd_itemsets = generate_potentially_discriminated_itemsets(train_data, ['derived_race'])
#     empty_pd_itemset = PD_itemset({})
#     pd_itemsets.append(empty_pd_itemset)
#
#     ubac = UBAC(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
#     ubac.fit(train_data)
#
#     scross = SCross(base_classifier=base_classifier, coverage=coverage)
#     scross.fit(train_data)
#
#     schreuder = SelectiveClassifierSchreuder(sensitive_attribute="derived_race", reject_rate_per_sens_group={1: coverage, 0:coverage}, base_classifier=base_classifier)
#     schreuder.fit(train_data)
#
#     new_ifac = IFAC_Alt_NoFlip(coverage=coverage, sensitive_attributes=['derived_race'],
#                                    reference_group_list=[PD_itemset({'derived_race': 'White'})], val1_ratio=val_ratio, val2_ratio=val_ratio,
#                                    base_classifier=base_classifier)
#     new_ifac.fit(train_data)
#
#
#     all_performances = pd.DataFrame([])
#     iteration = 1
#     for test_data in test_data_array:
#         iteration_performances = run_all_baselines(test_data, pd_itemsets,
#                                                    {"PlugIn": ubac, "SCross": scross, "Schreuder": schreuder, "IFAC": new_ifac},
#                                                    text_file = text_file, iteration=iteration)
#         all_performances = pd.concat([all_performances] + iteration_performances, ignore_index=True)
#         iteration += 1
#
#     averaged_performances = average_performance_results_over_multiple_splits(all_performances)
#     fairness_measures_over_average_performances = calculate_fairness_measures_over_averaged_performance_dataframe(averaged_performances)
#
#     averaged_performances.to_excel(path + "\\performances.xlsx")
#     fairness_measures_over_average_performances.to_excel(path + "\\fairness.xlsx")
#
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,"Positive Dec. Ratio", path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR", path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR", path_to_save_figure=path)
#
#
# def compare_recidivism(coverage, base_classifier, name_test_run):
#     recidivism_data = load_recidivism()
#     path = os.getcwd()
#     parent_dir = os.path.dirname(path) + r"\IFAC\results\recidivism_prediction"
#     name_test_run = "cov = " + str(coverage) + ", bb = " + str(base_classifier) + name_test_run
#     path = os.path.join(parent_dir, name_test_run)
#     os.mkdir(path)
#     text_file = open(path + "\\info.txt" , 'w')
#
#     train_ratio = 0.7
#     val_ratio = 0.25
#     test_set_size = 0.5
#     # test_set_size = int(len(income_prediction_data.descriptive_data) * train_ratio * val_ratio)
#     # print(test_set_size)
#     train_data, test_data_array = recidivism_data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
#                                                                                                  number_of_test_sets=20,
#                                                                                                  size_of_each_test_set=test_set_size)
#
#     pd_itemsets = generate_potentially_discriminated_itemsets(train_data, ['race'])
#     pd_itemsets.append(PD_itemset({}))
#
#     ubac = UBAC(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
#     ubac.fit(train_data)
#     #
#     schreuder = SelectiveClassifierSchreuder(sensitive_attributes=['race'],
#                                              reference_groups=[{'race': 'Caucasian'}],
#                                              reject_rate_per_sens_group={1: coverage, 0: coverage},
#                                              base_classifier=base_classifier)
#     schreuder.fit(train_data)
#     #
#     scross = SCross(base_classifier, coverage=coverage)
#     scross.fit(train_data)
#
#     auc = AUCPlugIn(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
#     auc.fit(train_data)
#
#     ifac = IFAC_Alt_NoFlip(coverage=coverage, sensitive_attributes=['race'],
#                     reference_group_list=[PD_itemset({'race': 'Caucasian'})],
#                     val1_ratio=val_ratio, val2_ratio=val_ratio, base_classifier=base_classifier)
#     ifac.fit(train_data)
#
#     all_performances = pd.DataFrame([])
#     iteration = 1
#     for test_data in test_data_array:
#         print("Iteration: " + str(iteration))
#         iteration_performances = run_all_baselines(test_data, pd_itemsets,
#                                                    {"PlugIn": ubac, "AUC": auc, "SCross": scross,
#                                                     "Schreuder": schreuder, "IFAC": ifac}, iteration=iteration,
#                                                    text_file=text_file)
#         all_performances = pd.concat([all_performances] + iteration_performances, ignore_index=True)
#         iteration += 1
#
#     averaged_performances = average_performance_results_over_multiple_splits(all_performances)
#     fairness_measures_over_average_performances = calculate_fairness_measures_over_averaged_performance_dataframe(averaged_performances)
#
#     averaged_performances.to_excel(path + "\\performances.xlsx")
#     fairness_measures_over_average_performances.to_excel(path + "\\fairness.xlsx")
#
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,"Positive Dec. Ratio", path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR", path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR", path_to_save_figure=path)
#
#
#
#
# def compare_grade_prediction(coverage, base_classifier, name_test_run):
#     grade_prediction_data = load_grade_prediction_data()
#     path = os.getcwd()
#     parent_dir = os.path.dirname(path) + r"\IFAC\grade_prediction"
#     name_test_run = "cov = " + str(coverage) + ", bb = " + str(base_classifier) + name_test_run
#     path = os.path.join(parent_dir, name_test_run)
#     os.mkdir(path)
#     text_file = open(path + "\\info.txt", 'w')
#
#     train_data, test_data_array = grade_prediction_data.split_into_train_and_multiple_test_sets(train_size=0.8,
#                                                                                                 number_of_test_sets=2)
#     train_data.descriptive_data = train_data.descriptive_data.drop(columns=["GroundTruth"])
#     train_data.one_hot_encoded_data = train_data.one_hot_encoded_data.drop(columns=["GroundTruth"])
#
#     pd_itemsets = generate_potentially_discriminated_itemsets(train_data, ['sex'])
#
#     ubac = UBAC(coverage=coverage, val_ratio=0.15, base_classifier=base_classifier)
#     ubac.fit(train_data)
#
#     scross = SCross(base_classifier=base_classifier, coverage=coverage)
#     scross.fit(train_data)
#
#     schreuder = SelectiveClassifierSchreuder(sensitive_attribute="sex",
#                                              reject_rate_per_sens_group={1: coverage, 0: coverage},
#                                              base_classifier=base_classifier)
#     schreuder.fit(train_data)
#
#     new_ifac = IFAC_Alt_NoFlip(coverage=coverage, sensitive_attributes=['sex'],
#                         reference_group_list=[PD_itemset({'sex': 'F'})], val1_ratio=0.15, val2_ratio=0.1,
#                         base_classifier=base_classifier)
#     new_ifac.fit(train_data)
#
#
#     all_performances = pd.DataFrame([])
#     iteration = 1
#     for test_data in test_data_array:
#         test_data_ground_truth = test_data.descriptive_data["GroundTruth"]
#         test_data.descriptive_data = test_data.descriptive_data.drop(columns=["GroundTruth"])
#         test_data.one_hot_encoded_data = test_data.one_hot_encoded_data.drop(columns=["GroundTruth"])
#         test_data.descriptive_data[test_data.decision_attribute] = test_data_ground_truth
#         test_data.one_hot_encoded_data[test_data.decision_attribute] = test_data_ground_truth
#
#         print("Iteration: ", iteration)
#         iteration_performances = run_all_baselines(test_data, pd_itemsets,
#                                                    {"PlugIn": ubac, "SCross": scross, "Schreuder": schreuder, "IFAC": new_ifac},
#                                                    text_file = text_file, iteration=iteration)
#         all_performances = pd.concat([all_performances] + iteration_performances, ignore_index=True)
#         iteration += 1
#
#     averaged_performances = average_performance_results_over_multiple_splits(all_performances)
#     fairness_measures_over_average_performances = calculate_fairness_measures_over_averaged_performance_dataframe(
#         averaged_performances)
#
#     averaged_performances.to_excel(path + "\\performances.xlsx")
#     fairness_measures_over_average_performances.to_excel(path + "\\fairness.xlsx")
#
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,
#                                                                               "Positive Dec. Ratio",
#                                                                               path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR",
#                                                                               path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR",
#                                                                               path_to_save_figure=path)
#
# def compare_german_credit(coverage, base_classifier, name_test_run):
#     credit_data = load_german_credit()
#     path = os.getcwd()
#     parent_dir = os.path.dirname(path) + r"\IFAC\results\german_credit"
#     name_test_run = "cov = " + str(coverage) + ", bb = " + str(base_classifier) + name_test_run
#     path = os.path.join(parent_dir, name_test_run)
#     os.mkdir(path)
#     text_file = open(path + "\\info.txt", 'w')
#
#     train_ratio = 0.8
#     val_ratio = 0.15
#     test_set_size = int(len(credit_data.descriptive_data) * train_ratio * val_ratio)
#     train_data, test_data_array = credit_data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
#                                                                                       number_of_test_sets=5, size_of_each_test_set=test_set_size)
#     pd_itemsets = generate_potentially_discriminated_itemsets(train_data, ['Sex'])
#     empty_pd_itemset = PD_itemset({})
#     pd_itemsets.append(empty_pd_itemset)
#
#     ubac = UBAC(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
#     ubac.fit(train_data)
#
#     scross = SCross(base_classifier=base_classifier, coverage=coverage)
#     scross.fit(train_data)
#
#     schreuder = SelectiveClassifierSchreuder(sensitive_attribute="Sex",
#                                              reject_rate_per_sens_group={1: coverage, 0: coverage},
#                                              base_classifier=base_classifier)
#     schreuder.fit(train_data)
#
#     ifac = IFAC_Alt_NoFlip(coverage=coverage, sensitive_attributes=['Sex'],
#                         reference_group_list=[PD_itemset({'Sex': 'Male'})], val1_ratio=val_ratio, val2_ratio=val_ratio,
#                         base_classifier=base_classifier)
#     ifac.fit(train_data)
#
#     all_performances = pd.DataFrame([])
#     iteration = 1
#     for test_data in test_data_array:
#         iteration_performances = run_all_baselines(test_data, pd_itemsets,
#                                                    {"PlugIn": ubac, "SCross": scross, "Schreuder": schreuder, "IFAC": ifac},
#                                                    text_file=text_file, iteration=iteration)
#         all_performances = pd.concat([all_performances] + iteration_performances, ignore_index=True)
#         iteration += 1
#
#     averaged_performances = average_performance_results_over_multiple_splits(all_performances)
#     fairness_measures_over_average_performances = calculate_fairness_measures_over_averaged_performance_dataframe(
#         averaged_performances)
#
#     averaged_performances.to_excel(path + "\\performances.xlsx")
#     fairness_measures_over_average_performances.to_excel(path + "\\fairness.xlsx")
#
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,
#                                                                               "Positive Dec. Ratio",
#                                                                               path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR",
#                                                                               path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR",
#                                                                               path_to_save_figure=path)
#
# def compare_census(coverage, base_classifier, name_test_run):
#     census_data = load_census_data()
#
#     path = os.getcwd()
#     parent_dir = os.path.dirname(path) + r"\IFAC\results\census"
#     name_test_run = "cov = " + str(coverage) + ", bb = " + str(base_classifier) + name_test_run
#     path = os.path.join(parent_dir, name_test_run)
#     os.mkdir(path)
#     text_file = open(path + "\\info.txt", 'w')
#
#     train_ratio = 0.7
#     val_ratio = 0.25
#     test_set_size = 0.5
#
#     train_data, test_data_array = census_data.split_into_train_and_multiple_test_sets(train_size=train_ratio,
#                                                                                       number_of_test_sets=20, size_of_each_test_set=test_set_size)
#     pd_itemsets = generate_potentially_discriminated_itemsets(train_data, ['sex'])
#     empty_pd_itemset = PD_itemset({})
#     pd_itemsets.append(empty_pd_itemset)
#
#     ubac = UBAC(coverage=coverage, val_ratio=val_ratio, base_classifier=base_classifier)
#     ubac.fit(train_data)
#
#     scross = SCross(base_classifier=base_classifier, coverage=coverage)
#     scross.fit(train_data)
#
#     auc = AUCPlugIn(coverage=coverage, val_ratio = val_ratio, base_classifier=base_classifier)
#     auc.fit(train_data)
#
#     schreuder = SelectiveClassifierSchreuder(sensitive_attributes=['sex'],
#                                              reference_groups=[{'sex': 'Male'}],
#                                              reject_rate_per_sens_group={1: coverage, 0: coverage},
#                                              base_classifier=base_classifier)
#     schreuder.fit(train_data)
#
#     ifac = IFAC_Alt_NoFlip(coverage=coverage, sensitive_attributes=['sex'],
#                         reference_group_list=[PD_itemset({'sex': 'Male'})], val1_ratio=val_ratio, val2_ratio=val_ratio,
#                         base_classifier=base_classifier)
#     ifac.fit(train_data)
#
#     all_performances = pd.DataFrame([])
#     iteration = 1
#     for test_data in test_data_array:
#         iteration_performances = run_all_baselines(test_data, pd_itemsets,
#                                                    {"PlugIn": ubac, "AUC": auc,
#                                                     "SCross": scross,
#                                                     "Schreuder": schreuder, "IFAC": ifac},
#                                                    iteration=iteration,
#                                                    text_file=text_file)
#         all_performances = pd.concat([all_performances] + iteration_performances, ignore_index=True)
#         iteration += 1
#
#     averaged_performances = average_performance_results_over_multiple_splits(all_performances)
#     fairness_measures_over_average_performances = calculate_fairness_measures_over_averaged_performance_dataframe(
#         averaged_performances)
#
#     averaged_performances.to_excel(path + "\\performances.xlsx")
#     fairness_measures_over_average_performances.to_excel(path + "\\fairness.xlsx")
#
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,
#                                                                               "Positive Dec. Ratio",
#                                                                               path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR",
#                                                                               path_to_save_figure=path)
#     visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR",
#                                                                               path_to_save_figure=path)
#
#
