from experiments import run_experiment, save_for_gui, test_schreuder_stuff, run_baseline_classifier
from preprocessing_scripts.prepare_census_income import prepare_census_income
from preprocessing_scripts.prepare_mortgate_dataset import prepare_mortgage_data
from load_datasets import load_income_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_experiment(task="recidivism", coverage=0.8, base_classifier='Random Forest', name_test_run="GUI")
    # run_experiment(task="recidivism", coverage=0.8, base_classifier='Random Forest', name_test_run="multipleIFACS")
    # run_experiment(task="recidivism", coverage=0.9, base_classifier='Random Forest', name_test_run="multipleIFACS")
    # run_experiment(task="recidivism", coverage=0.99, base_classifier='Random Forest', name_test_run="multipleIFACS")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
