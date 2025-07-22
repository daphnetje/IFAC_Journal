import pandas as pd
from plotnine import *
from mizani.palettes import hue_pal
from pandas.api.types import CategoricalDtype


def convert_to_nice_num_format(some_float):
    rounded_by_three = format(some_float, '.2f')
    string_to_float = str(rounded_by_three)
    after_0 = string_to_float.split(".")[1]
    output = "." + after_0
    return output

def results_to_latex_table():
    dataset_sens_dict = {"census": "sex", "income": "sex, race", "oulad": "disability"}
    decision_tasks = ["income", "oulad", "census"]
    selective_classifier_names = ["PlugIn", "SCross", "AUC", "Schreuder", "IFAC"]
    coverages = [0.99, 0.9, 0.8, 0.7]

    performance_df = pd.read_excel("final_results\\census\cov = 0.7, bb = Random Forest\performances.xlsx")
    fairness_df = pd.read_excel("final_results\\census\cov = 0.7, bb = Random Forest\\fairness.xlsx")

    performance_df['Group'] = performance_df['Group'].fillna("")

    coverages = ""
    accuracies = ""
    fairnesses = ""
    i = 0

    for selective_classifier in selective_classifier_names:
        relevant_performance_row = performance_df[
            (performance_df["Classification Type"] == selective_classifier) & (performance_df["Group"] == "")]
        relevant_fairness_row = fairness_df[
            (fairness_df["Classification Type"] == selective_classifier) & (fairness_df["Sensitive Features"] == "sex")]

        coverage = relevant_performance_row["Coverage mean"].item()
        coverage_ci = relevant_performance_row["Coverage ci"].item()
        coverage_entry = convert_to_nice_num_format(coverage) + " $\pm$ " + convert_to_nice_num_format(coverage_ci)
        coverages += (coverage_entry)

        accuracy = relevant_performance_row["Accuracy mean"].item()
        accuracy_ci = relevant_performance_row["Accuracy ci"].item()
        accuracy_entry = convert_to_nice_num_format(accuracy) + " $\pm$ " + convert_to_nice_num_format(accuracy_ci)
        accuracies += (accuracy_entry)

        fairness = relevant_fairness_row["Highest Diff. in Pos. Ratio mean"].item()
        fairness_ci = relevant_fairness_row["Highest Diff. in Pos. Ratio ci"].item()
        fairness_entry = convert_to_nice_num_format(fairness) + " $\pm$ " + convert_to_nice_num_format(fairness_ci)
        fairnesses += fairness_entry

        i += 1
        if (i != len(selective_classifier_names)):
            coverages += " & "
            accuracies += " & "
            fairnesses += " & "

    print(coverages)
    print(accuracies)
    print(fairnesses)

    row = coverages + " & " + accuracies + " & " + fairnesses
    print(row)

def custom_labeller(metric):
    metric_names = {'emp_cov': 'Empirical Coverage \u2501', 'acc': 'Accuracy \u2191', 'group fairness': 'Group Unfairness \u2193', 'indiv. fairness': 'Individual Unfairness \u2193'}
    return metric_names[metric]

def summary_results_to_pretty_chart_plots(dataset, classifier, baseline_acc, baseline_group_fair, baseline_indiv_fair):
    #FILL IN HERE THE ACCURACY AND UNFAIRNESS OF A NON-SELECTIVE CLASSIFIER ON THE SPECIFIED DATASET
    baseline_df = pd.DataFrame({
        'metric': ['acc', 'group fairness', 'indiv. fairness'],
        'baseline': [baseline_acc, baseline_group_fair, baseline_indiv_fair]
    })

    df = pd.read_excel("final_results\\"+dataset+"\\"+classifier+"with_individual_fairness-summary.xlsx") # or pd.read_csv() if needed

    # Reshape the data for easier faceting
    df_long = pd.concat([
        df.assign(metric='emp_cov', value=df['emp_cov'], ci=df['emp_cov_ci']),
        df.assign(metric='acc', value=df['acc'], ci=df['acc_ci']),
        df.assign(metric='group fairness', value=df['group fairness'], ci=df['group fairness_ci']),
        df.assign(metric='indiv. fairness', value=df['indiv. fairness'], ci=df['indiv. fairness_ci'])
    ])

    metric_order = CategoricalDtype(categories=['emp_cov', 'acc', 'group fairness', 'indiv. fairness'], ordered=True)
    df_long['metric'] = df_long['metric'].astype(metric_order)
    baseline_df['metric'] =  baseline_df['metric'].astype(metric_order)

    methods_order = CategoricalDtype(categories=['PlugIn', 'SCross', 'AUC', 'DPWA', 'IFAC'], ordered=True)
    df_long['method'] = df_long['method'].astype(methods_order)
    df_long['coverage_str'] = df_long['coverage'].astype(str)

    # Custom color palette
    methods = df['method'].unique()
    colors = hue_pal()(len(methods))
    print(colors)

    dodge = position_dodge(width=0.7)

    p = (
            ggplot(df_long, aes(x='coverage_str', y='value', group='method', colour='method'))
            + geom_hline(data=baseline_df, mapping=aes(yintercept='baseline'), linetype='dashed', color='#3A3B3C',
                         size=0.5)
            + geom_line(position=dodge, alpha=0.85, size=1, show_legend=True)
            + geom_point(position=dodge, size=2, shape='o', stroke=1, show_legend=True)
            + geom_errorbar(aes(ymin='value - ci', ymax='value + ci'),
                position=dodge, width=0.2)
            + facet_wrap('~metric', scales='fixed', nrow=1, labeller=labeller(cols=custom_labeller))
            + labs(x='Coverage', y='Value', color='Method')
            + theme_minimal()
            + theme(
                figure_size=(14, 5),
                axis_line=element_line(color='#3A3B3C', size=1),
                axis_text_x=element_text(size=11, hjust=0.5, vjust=0.01, margin={'t': 0}, weight='bold'),
                axis_text_y=element_text(size=11, weight='bold'), #rotation = 0.45
                axis_title_x = element_text(weight='bold'),
                axis_title_y = element_text(weight='bold'),
                strip_text=element_text(size=11, weight='bold'),
                plot_title=element_text(size=15, weight='bold', ha='center'),
                legend_title=element_text(size=11, weight='bold'),
                legend_text=element_text(size=11),
                legend_position='right',
                panel_spacing = 0.5,
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank(),
                panel_grid_minor_y=element_blank())
            + scale_colour_manual(values={"PlugIn": "#db5f57", "AUC": "#b9db57", "IFAC": "#57db94", "DPWA": "#5784db", "SCross": "#c957db"})
            + scale_y_continuous(breaks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

    print(p)

def summary_results_to_pretty_bar_plots(dataset, classifier, baseline_acc, baseline_fair):
    #FILL IN HERE THE ACCURACY AND UNFAIRNESS OF A NON-SELECTIVE CLASSIFIER ON THE SPECIFIED DATASET
    baseline_df = pd.DataFrame({
        'metric': ['acc', 'group fairness'],
        'baseline': [baseline_acc, baseline_fair]
    })

    df = pd.read_excel("final_results\\"+dataset+"\\"+classifier+"with_individual_fairness-summary.xlsx") # or pd.read_csv() if needed

    # Reshape the data for easier faceting
    df_long = pd.concat([
        df.assign(metric='emp_cov', value=df['emp_cov'], ci=df['emp_cov_ci']),
        df.assign(metric='acc', value=df['acc'], ci=df['acc_ci']),
        df.assign(metric='group fairness', value=df['group fairness'], ci=df['group fairness_ci']),
        df.assign(metric='indiv. fairness', value=df['indiv. fairness'], ci=df['indiv. fairness_ci'])
    ])

    metric_order = CategoricalDtype(categories=['emp_cov', 'acc', 'group fairness', 'indiv. fairness'], ordered=True)
    df_long['metric'] = df_long['metric'].astype(metric_order)
    baseline_df['metric'] =  baseline_df['metric'].astype(metric_order)

    methods_order = CategoricalDtype(categories=['PlugIn', 'SCross', 'AUC', 'DPWA', 'IFAC'], ordered=True)
    df_long['method'] = df_long['method'].astype(methods_order)
    df_long['coverage_str'] = df_long['coverage'].astype(str)

    # Custom color palette
    methods = df['method'].unique()
    colors = hue_pal()(len(methods))
    print(colors)

    p = (
            ggplot(df_long, aes(x='coverage_str', y='value', fill='method', group='method', colour='method'))
            + geom_hline(data=baseline_df, mapping=aes(yintercept='baseline'), linetype='dashed', color='#3A3B3C',
                         size=0.5)
            + geom_col(position=position_dodge(width=0.8), width=0.7, alpha=0.85)
            + geom_errorbar(aes(ymin='value - ci', ymax='value + ci'),
                position=position_dodge(width=0.8), width=0.15, colour="black")
            + facet_wrap('~metric', scales='fixed', nrow=1, labeller=labeller(cols=custom_labeller))
            + labs(x='Coverage', y='Value', color='Method')
            + theme_minimal()
            + theme(
                figure_size=(14, 4),
                axis_text_x=element_text(size=11, hjust=0.5, vjust=0.01, margin={'t': 0}, weight='bold'),
                axis_text_y=element_text(size=11, weight='bold'), #rotation = 0.45
                axis_title_x = element_text(weight='bold'),
                axis_title_y = element_text(weight='bold'),
                strip_text=element_text(size=11, weight='bold'),
                plot_title=element_text(size=15, weight='bold', ha='center'),
                legend_title=element_text(size=11, weight='bold'),
                legend_text=element_text(size=11),
                legend_position='right',
                panel_spacing = 0.5,
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank(),
                panel_grid_minor_y=element_blank())
            + scale_fill_manual(values={"PlugIn": "#db5f57", "AUC": "#b9db57", "IFAC": "#57db94", "DPWA": "#5784db", "SCross": "#c957db"},
                                labels=['PlugIn', 'SCross', 'AUC', 'DPWA', 'IFAC'])
            + scale_colour_manual(values={"PlugIn": "#db5f57", "AUC": "#b9db57", "IFAC": "black", "DPWA": "#5784db", "SCross": "#c957db"}, guide=False)
            + scale_y_continuous(breaks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

    print(p)


def summarize_results_per_dataset(dataset, classifier):
    dataset_sens_dict = {"census": "sex", "income": "sex, race", "oulad": "disability", "mortgage": "derived_race", "recidivism" : "race"}
    selective_classifier_names = ["PlugIn", "SCross", "AUC", "Schreuder", "IFAC"]
    coverages = ["0.7", "0.8", "0.9", "0.99"]
    performance_entries = []

    sensitive_attributes = dataset_sens_dict[dataset]

    for cov in coverages:
        performance_df = pd.read_excel("final_results\\" + dataset + "\cov = " + cov + ", bb = "+ classifier + "debug3\\performances.xlsx", engine='openpyxl')
        fairness_df = pd.read_excel("final_results\\" + dataset + "\cov = " + cov + ", bb = "+ classifier + "debug3\\fairness.xlsx", engine='openpyxl')

        performance_df = performance_df.fillna("")
        fairness_df = fairness_df.fillna("")

        for selective_classifier in selective_classifier_names:
            relevant_performance_row = performance_df[
                (performance_df["Classification Type"] == selective_classifier) & (performance_df["Group"] == "")]
            relevant_fairness_row = fairness_df[
                (fairness_df["Classification Type"] == selective_classifier) & (
                            fairness_df["Sensitive Features"] == sensitive_attributes)]

            coverage = relevant_performance_row["Coverage mean"].item()
            coverage_ci = relevant_performance_row["Coverage ci"].item()

            accuracy = relevant_performance_row["Accuracy mean"].item()
            accuracy_ci = relevant_performance_row["Accuracy ci"].item()

            situation_testing = relevant_performance_row["Avg. Situation Testing Non-Rejected mean"].item()
            situation_testing_ci = relevant_performance_row["Avg. Situation Testing Non-Rejected ci"].item()

            fairness = relevant_fairness_row["Highest Diff. in Pos. Ratio mean"].item()
            fairness_ci = relevant_fairness_row["Highest Diff. in Pos. Ratio ci"].item()

            if selective_classifier == "Schreuder":
                selective_classifier = "DPWA"

            performance_entry = {"method": selective_classifier, "coverage": float(cov), "emp_cov": coverage, "emp_cov_ci": coverage_ci, "acc": accuracy, "acc_ci": accuracy_ci, "group fairness": fairness, "group fairness_ci": fairness_ci, "indiv. fairness": situation_testing, "indiv. fairness_ci": situation_testing_ci}
            performance_entries.append(performance_entry)
    performance_df = pd.DataFrame(performance_entries)
    performance_df.to_excel("final_results\\"+dataset+"\\"+classifier+"with_individual_fairness-summary.xlsx")


if __name__ == '__main__':
    summarize_results_per_dataset("recidivism", "Random Forest")
    summary_results_to_pretty_chart_plots("recidivism", "Random Forest", 0.62, 0.10, 0.08)

    # summarize_results_per_dataset("census", "NN")
    # summary_results_to_pretty_plots("census", "NN", 0.80, 0.48)

