import pandas as pd


def generate_file_for_coverage(cov):
    tasks = ["census", "income", "mortgage", "oulad", "recidivism"]

    datasets = []
    for task in tasks:
        NN_df = pd.read_excel("final_results\\" + task + "\\NN-summary.xlsx",engine='openpyxl')
        NN_df["task"] = "NN"
        # NN_df["method"] = NN_df["method"] + "_NN"

        XGB_df = pd.read_excel("final_results\\" + task + "\\XGB-summary.xlsx",engine='openpyxl')
        XGB_df["task"] = "XGB"
        # XGB_df["method"] = XGB_df["method"] + "_XGB"

        RF_df = pd.read_excel("final_results\\" + task + "\\RF-summary.xlsx",engine='openpyxl')
        RF_df["task"] = "RF"
        # RF_df["method"] = RF_df["method"] + "_RF"

        stacked_dfs = pd.concat([NN_df, XGB_df, RF_df])
        print(stacked_dfs)
        relevant_cov_performance_and_fairness = stacked_dfs[stacked_dfs["coverage"]==cov][["method", "acc", "fairness", "task"]]
        relevant_cov_performance_and_fairness["task"] += task
        datasets.append(relevant_cov_performance_and_fairness)

    all_datasets = pd.concat(datasets).reset_index()
    print(all_datasets)
    all_datasets.to_csv("coverage="+str(cov)+".csv")

generate_file_for_coverage(0.7)
generate_file_for_coverage(0.8)
generate_file_for_coverage(0.9)
generate_file_for_coverage(0.99)