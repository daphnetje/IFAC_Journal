from experiments import run_experiment, save_for_gui, test_schreuder_stuff, run_baseline_classifier
from preprocessing_scripts.prepare_census_income import prepare_census_income
from preprocessing_scripts.prepare_mortgate_dataset import prepare_mortgage_data
from load_datasets import load_income_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # run_baseline_classifier(task="mortgage", base_classifier="Random Forest")
    # run_baseline_classifier(task="mortgage", base_classifier="XGB")
    # run_baseline_classifier(task="mortgage", base_classifier="NN")
    #test_schreuder_stuff(coverage=0.8, base_classifier="Random Forest")
    #save_for_gui(coverage=0.8, task="recidivism", base_classifier="Random Forest")
    # run_experiment(task="census", coverage=0.99, base_classifier='NN', name_test_run="debug3")
    # run_experiment(task="census", coverage=0.9, base_classifier='NN', name_test_run="debug3")
    # run_experiment(task="census", coverage=0.8, base_classifier='NN', name_test_run="debug3")
    # run_experiment(task="census", coverage=0.7, base_classifier='NN', name_test_run="debug3")
    #
    # run_experiment(task="recidivism", coverage=0.99, base_classifier='NN', name_test_run="debug3")
    # run_experiment(task="recidivism", coverage=0.9, base_classifier='NN', name_test_run="debug3")
    # run_experiment(task="recidivism", coverage=0.8, base_classifier='NN', name_test_run="debug3")


    # run_experiment(task="oulad", coverage=0.7, base_classifier='XGB', name_test_run="debug3")
    # run_experiment(task="oulad", coverage=0.8, base_classifier='XGB', name_test_run="debug3")
    # run_experiment(task="oulad", coverage=0.9, base_classifier='XGB', name_test_run="debug3")
    # run_experiment(task="oulad", coverage=0.99, base_classifier='XGB', name_test_run="debug3")

    run_experiment(task="mortgage", coverage=0.7, base_classifier='Random Forest', name_test_run="multipleIFACS")
    run_experiment(task="mortgage", coverage=0.8, base_classifier='Random Forest', name_test_run="multipleIFACS")
    run_experiment(task="mortgage", coverage=0.9, base_classifier='Random Forest', name_test_run="multipleIFACS")
    run_experiment(task="mortgage", coverage=0.99, base_classifier='Random Forest', name_test_run="multipleIFACS")
    # run_experiment(task="mortgage", coverage=0.8, base_classifier='XGB', name_test_run="debug3")
    # run_experiment(task="mortgage", coverage=0.9, base_classifier='XGB', name_test_run="debug3")
    # run_experiment(task="mortgage", coverage=0.99, base_classifier='XGB', name_test_run="debug3")




    #run_experiment(task="mortgage", coverage=0.8, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    #run_experiment(task="census", coverage=0.8, base_classifier='Random Forest', name_test_run=" test_individual_fairness")


    # run_experiment(task="census", coverage=0.99, base_classifier='NN', name_test_run="")
    # run_experiment(task="census", coverage=0.9, base_classifier='NN', name_test_run="")
    # run_experiment(task="census", coverage=0.7, base_classifier='NN', name_test_run="")
    # #
    # run_experiment(task="mortgage", coverage=0.99, base_classifier='NN', name_test_run="")
    # run_experiment(task="mortgage", coverage=0.9, base_classifier='NN', name_test_run="")
    # run_experiment(task="mortgage", coverage=0.7, base_classifier='NN', name_test_run="")
    #
    #run_experiment(task="oulad", coverage=0.8, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    #run_experiment(task="recidivism", coverage=0.8, base_classifier='Random Forest', name_test_run="test_individual_fairness")


    # run_experiment(task="income", coverage=0.99, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    # run_experiment(task="income", coverage=0.9, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    # run_experiment(task="income", coverage=0.8, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    #run_experiment(task="income", coverage=0.7, base_classifier='Random Forest', name_test_run="test_individual_fairness")

    #run_experiment(task="oulad", coverage=0.99, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    # run_experiment(task="oulad", coverage=0.9, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    # run_experiment(task="oulad", coverage=0.7, base_classifier='Random Forest', name_test_run="test_individual_fairness")

    # run_experiment(task="recidivism", coverage=0.7, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    # run_experiment(task="recidivism", coverage=0.8, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    # run_experiment(task="recidivism", coverage=0.9, base_classifier='Random Forest', name_test_run="test_individual_fairness")
    # run_experiment(task="recidivism", coverage=0.99, base_classifier='Random Forest', name_test_run="test_individual_fairness")

    # run_experiment(task="recidivism", coverage=0.99, base_classifier='Random Forest',
    #                name_test_run="debug3")

    # run_experiment(task="oulad", coverage=0.7, base_classifier='Random Forest',
    #                name_test_run="debug3")
    # run_experiment(task="oulad", coverage=0.8, base_classifier='Random Forest',
    #                name_test_run="debug3")
    # run_experiment(task="oulad", coverage=0.9, base_classifier='Random Forest',
    #                name_test_run="debug3")
    # run_experiment(task="oulad", coverage=0.99, base_classifier='Random Forest',
    #                name_test_run="debug3")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
