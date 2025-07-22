from folktables import ACSDataSource, ACSEmployment, ACSPublicCoverage
import folktables
import pandas as pd

ACS_categories = {
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
    "MIL": {
        1.0: "Now on active duty",
        2.0: "On active duty in the past, but not now",
        3.0: "Only on active duty for training in Reserves/National Guard",
        4.0: "Never served in the military",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married",
    },
    "DIS": {1.0: "With a disability", 2.0: "Without a disability"},
    "CIT": {
        1.0: "Born in the United States",
        2.0: "Born in Puerto Rico, Guam, the U.S. Virgin Islands or Northern Marianas",
        3.0: "Born abroad of U.S. citizen parent or parents",
        4.0: "U.S. citizen by naturalization",
        5.0: "Not a U.S. citizen",
    },
    "DEAR": {   #hearing difficulty
        1.0: "Yes",
        2.0: "No",
    },
    "DEYE": {  #vision difficulty
        1.0: "Yes",
        2.0: "No",
    },
    "DREM": {  #cognitive difficulty
        "nan": "N/A (less than 5 years old)",
        1.0: "Yes",
        2.0: "No"
    },
    "FER": {   #give birth to child within the past 12 months
        "nan": "N/A (younger than 15, older than 50 and/or male)",
        1.0: "Yes",
        2.0: "No",
    },
    "OC": { #owns childs
        0.0: "No",
        1.0: "Yes"
    }

}

def change_employment_label_names(row):
    if row['work status']:
        return "employed"
    else:
        return "unemployed"


def change_income_label_names(row):
    if row['income']:
        return "high"
    else:
        return "low"


def change_insurance_coverage_label_names(row):
    if row['insurance coverage']:
        return "with coverage"
    else:
        return "without coverage"


def bin_workclass(row):
    if row['workclass'] in [1.0, 2.0]:
        return "private"
    if row['workclass'] in [3.0, 4.0, 5.0]:
        return "governmental"
    if row['workclass'] in [6.0, 7.0]:
        return "self employed"
    return "no paid work"


def bin_marital_status(row):
    if row['marital status'] in ['Divorced', 'Separated']:
        return 'Separated'
    else:
        return row['marital status']


# def bin_race(row):
#     if row['race'] in ['American Indian and Alaska Native tribes specified;or American Indian or Alaska Native,not specified and no other', 'Alaska Native alone', 'American Indian alone']:
#         return "American Indian or Alaska Native"
#     if row['race'] in ['Some Other Race alone', 'Two or More Races']:
#         return 'One or More Other Races'
#     return row['race']

def bin_race(row):
    if row['race'] in ['Black or African American alone', 'White alone']:
        return row['race']
    else:
        return 'Other'


def bin_education(raw_data):
    # 1/2/3 - No elementary school
    # 4-8 - Elementary School
    # 9-11 - Middle School
    # 12-15 - Started High School, no diploma
    # 16/17 - High School or GED
    # 18/19 - Started College, no degree
    # 20 - Associates Degree
    # 21 - Bachelors Degree
    # 22/23 - Master or other Degree beyond Bachelor
    # 24 - Doctorate Degree
    cut_labels_education = ["No Elementary School", "Elementary School", "Middle School",
                            "Started High School, No Diploma", "High School or GED Diploma",
                            "Started College, No Diploma", "Associate Degree", "Bachelor Degree",
                            "Master or other Degree Beyond Bachelor", "Doctorate Degree"]
    cut_labels_education_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cut_bins_education = [0, 3, 8, 11, 15, 17, 19, 20, 21, 23, 24]
    raw_data['education_num'] = pd.cut(raw_data['education'], bins=cut_bins_education,
                                       labels=cut_labels_education_num, right=True)
    raw_data['education'] = pd.cut(raw_data['education'], bins=cut_bins_education, labels=cut_labels_education, right=True)
    return raw_data


def bin_age(raw_data):
    cut_labels_age = ["Younger than 25", "25-29", "30-39", "40-49", "50-59", "60-69", "Older than 70"]
    cut_labels_age_num = [1, 2, 3, 4, 5, 6, 7]
    cut_bins_age = [0, 24, 29, 39, 49, 59, 69, 100]
    raw_data['age_num'] = pd.cut(raw_data['age'], bins=cut_bins_age, labels=cut_labels_age_num)
    raw_data['age'] = pd.cut(raw_data['age'], bins=cut_bins_age, labels=cut_labels_age)
    return raw_data


def bin_workinghours(raw_data):
    cut_labels_workhours = ["Less than 20", "20-39", "40-49", "More than 50"]
    cut_labels_workhours_num = [1, 2, 3, 4]
    cut_bins_workhours = [0, 19, 39, 49, 100]
    raw_data['workinghours_num'] = pd.cut(raw_data['workinghours'], bins=cut_bins_workhours,
                                            labels=cut_labels_workhours_num)
    raw_data['workinghours'] = pd.cut(raw_data['workinghours'], bins=cut_bins_workhours,
                                        labels=cut_labels_workhours)

    return raw_data


def bin_income(raw_data):
    raw_data['income_binned'] = pd.cut(raw_data['income'], bins=5)
    raw_data['income_binned_num'] = pd.cut(raw_data['income'], bins=5, labels = [1, 2, 3, 4, 5])
    return raw_data


def bin_occupation(row):
    occupation_mapping = {
        range(0, 751): "Management/Business",
        range(800, 961): "Finance/Accounting",
        range(1005, 1981): "Science, Engineering, Technology",
        range(2001, 2061): "Counseling/Mental Health Services",
        range(2100, 2181): "Legal Services",
        range(2205, 2556): "Education",
        range(2600, 2921): "Entertainment",
        range(3000, 3656): "Healthcare/Medical Services",
        range(3700, 3971): "Protective Services",
        range(4000, 4656): "Service/Hospitality",
        range(4700, 4966): "Sales",
        range(5000, 5941): "Office/Administrative Support",
        range(6005, 6131): "Farming, Fishing, Forestry",
        range(6200, 6951): "Construction/Extraction",
        range(7000, 7641): "Repair/Maintenance",
        range(7700, 8991): "Production/Assembly",
        range(9005, 9761): "Transport",
        range(9800, 9831): "Military Services",
        range(9920, 9921): "Unemployed"
    }

    for key, value in occupation_mapping.items():
        if row['occupation'] in key:
            return value

    return "Unemployed"


def prepare_income_prediction_data(sample_size = 20000):
    #'AGEP','COW', 'SCHL', 'MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P',
    feature_names = ['AGEP','COW', 'SCHL', 'MAR','OCCP','WKHP','SEX','RAC1P', 'ENG', 'FER']
    renamed_features_dict = {'AGEP': 'age', 'COW': 'workclass', 'SCHL': 'education',
                             'MAR': 'marital status', 'OCCP': 'occupation', 'POBP': 'place of birth',
                             'WKHP': 'workinghours', 'SEX': 'sex', 'RAC1P': 'race',
                             'ENG': 'ability to speak english', 'FER': 'gave birth this year'
                             }

    ACSIncomePredictionTask = folktables.BasicProblem(
        features=feature_names,
        target='PINCP',  #PINCP stands for total person's income
        target_transform=lambda x: x > 50000,
        group='RAC1P',
        preprocess=folktables.adult_filter,
    )

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    al_data = data_source.get_data(states=["AL"], download=True)
    raw_data, labels, _ = ACSIncomePredictionTask.df_to_pandas(al_data, categories=ACS_categories)
    raw_data['income'] = labels

    raw_data = raw_data.rename(columns=renamed_features_dict)

    raw_data = bin_education(raw_data)
    raw_data = bin_age(raw_data)
    raw_data = bin_workinghours(raw_data)

    raw_data['workclass'] = raw_data.apply(lambda  row: bin_workclass(row), axis=1)
    raw_data['marital status'] = raw_data.apply(lambda row: bin_marital_status(row), axis=1)
    raw_data['occupation'] = raw_data.apply(lambda row: bin_occupation(row), axis=1)
    raw_data['race'] = raw_data.apply(lambda row: bin_race(row), axis=1)
    raw_data['income'] = raw_data.apply(lambda row: change_income_label_names(row), axis=1)

    raw_data_sample = raw_data.sample(n=sample_size, random_state=7)
    numerical_and_descriptive_dataframe = raw_data_sample[
        ['age', 'age_num', 'marital status', 'workinghours', 'workinghours_num', 'education', 'education_num', 'workclass', 'occupation', 'race',
         'sex', 'income']]
    numerical_and_descriptive_dataframe.to_csv('income_sample.csv')

    return numerical_and_descriptive_dataframe


