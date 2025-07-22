import pandas as pd

def bin_race(row):
    if row['race'] not in ['Caucasian', 'African American']:
        return 'Other'
    else:
        return row['race']


def bin_offense_type(row):
    if row['offense'] not in ["OAR/OAS", "Operating While Intoxicated", "Operating Without License",
                                "Disorderly Conduct", "Battery", "Resisting Officer",
                                "Drug Posession", "Bail Jumping", "Burglary"]:

        return "Other"
    return row['offense']


def bin_age(data):
    cut_labels_age = ["Younger than 18", "18-29", "30-39", "40-49", "50-59", "Older than 60"]
    cut_labels_age_num = [1, 2, 3, 4, 5, 6]
    cut_bins_age = [0, 17, 29, 39, 49, 59, 100]
    data['age_num'] = pd.cut(data['age'], bins=cut_bins_age, labels=cut_labels_age_num)
    data['age'] = pd.cut(data['age'], bins=cut_bins_age, labels=cut_labels_age)
    return data

def bin_prior_criminal_count(data, respective_column):
    cut_labels_prior_criminal_count = ["None", "1-5", "6-10", "More than 10"]
    cut_labels_prior_criminal_count_num = [0, 1, 2, 3]
    cut_bins_prior_criminal_count = [-1, 0, 5, 10, 1000]
    numerical_column_name = respective_column + "_num"
    data[numerical_column_name] = pd.cut(data[respective_column], bins=cut_bins_prior_criminal_count, labels=cut_labels_prior_criminal_count_num)
    data[respective_column] = pd.cut(data[respective_column], bins=cut_bins_prior_criminal_count, labels=cut_labels_prior_criminal_count)
    return data


def change_recidivism_label_names(row):
    if row['recidivism']:
        return "yes"
    else:
        return "no"

def prepare_recidivism_data():
    data = pd.read_csv('wisconsin_criminal_cases.csv')
    data = data[data['recid_180d'].notna()]

    renamed_features_dict = {'recid_180d': 'recidivism', 'age_offense': 'age', 'wcisclass': 'offense'}

    data = data.rename(columns=renamed_features_dict)
    data = data[['sex', 'race', 'age', 'case_type', 'offense', 'prior_felony', 'prior_misdemeanor',
                 'prior_criminal_traffic', 'highest_severity', 'recidivism']]

    data = bin_age(data)
    data = bin_prior_criminal_count(data, 'prior_felony')
    data = bin_prior_criminal_count(data, 'prior_misdemeanor')
    data = bin_prior_criminal_count(data, 'prior_criminal_traffic')
    data['race'] = data.apply(lambda row: bin_race(row), axis=1)
    data['offense'] = data.apply(lambda row: bin_offense_type(row), axis=1)
    data['recidivism'] = data.apply(lambda row: change_recidivism_label_names(row), axis=1)

    descriptive_dataframe = data[
        ['race', 'age', 'case_type', 'offense', 'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic', 'recidivism']]
    descriptive_dataframe.to_csv('preprocessed_data/recidivism_data.csv')
    #categorical_features = ['case_type', 'offense', 'race']


