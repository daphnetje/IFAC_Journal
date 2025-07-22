import pandas as pd
import numpy as np

def bin_weeks_worked_per_year(raw_data):
    cut_labels_weeks = [">=26", "27-51", "=52"]
    cut_bins_weeks = [-1, 26, 51, 52]
    raw_data['weeks worked in year'] = pd.cut(raw_data['weeks worked in year'], bins=cut_bins_weeks,
                                              labels=cut_labels_weeks)
    return raw_data

def bin_capital_gain(raw_data):
    cut_labels_gain = ["<=500", ">500"]
    cut_bins_gain = [-1, 500, 100000000000000]
    raw_data['capital gain'] = pd.cut(raw_data['capital gain'], bins=cut_bins_gain,
                                      labels=cut_labels_gain)
    return raw_data

def bin_wage_per_hour(raw_data):
    cut_labels_wage = ["<500", "500-1000", "More than 1000"]
    cut_bins_wage = [-1, 499, 999, 1000000000000000]
    raw_data['wage per hour'] = pd.cut(raw_data['wage per hour'], bins=cut_bins_wage,
                                      labels=cut_labels_wage)
    return raw_data

def bin_capital_loss(raw_data):
    cut_labels_loss = ["<=500", ">500"]
    cut_bins_loss = [-1, 500, np.inf]
    raw_data['capital loss'] = pd.cut(raw_data['capital loss'], bins=cut_bins_loss,
                                      labels=cut_labels_loss)
    return raw_data

def bin_age(raw_data):
    cut_labels_age = ["Younger than 25", "26-60", "Older than 60"]
    cut_bins_age = [0, 25, 60, 100]
    raw_data['age'] = pd.cut(raw_data['age'], bins=cut_bins_age, labels=cut_labels_age)
    return raw_data



#  ' Less than 1st grade' ' 1st 2nd 3rd or 4th grade' ' 5th or 6th grade' \
#  ' 7th and 8th grade' ' 9th grade' ' 10th grade'  ' 11th grade'  ' 12th grade no diploma'
# ' High school graduate'
# ' Some college but no degree' ' Associates degree-academic program' ' Associates degree-occup /vocational'
# ' Bachelors degree(BA AB BS)'  ' Masters degree(MA MS MEng MEd MSW MBA)'
#  ' Prof school degree (MD DDS DVM LLB JD)'
# ' Doctorate degree(PhD EdD)'
def bin_education(row):
    if row['education'] in [' Less than 1st grade', ' 1st 2nd 3rd or 4th grade', ' 5th or 6th grade']:
        return "Elementary School"
    if row['education'] in [' 7th and 8th grade', ' 9th grade', ' 10th grade']:
        return "Middle School"
    if row['education'] in [' 11th grade',  ' 12th grade no diploma']:
        return "High School, no diploma"
    if row['education'] in [' High school graduate']:
        return "High School Degree"
    if row['education'] in [' Some college but no degree', ' Associates degree-academic program', ' Associates degree-occup /vocational']:
        return "College or Associate"
    if row['education'] in [' Bachelors degree(BA AB BS)', ' Masters degree(MA MS MEng MEd MSW MBA)']:
        return "University Degree"
    if row['education'] in [' Doctorate degree(PhD EdD)', ' Prof school degree (MD DDS DVM LLB JD)']:
        return "Professor/Doctorate"


def prepare_census_income():
    raw_data = pd.read_csv('preprocessing_scripts/census-income.csv')
    raw_data.dropna(inplace=True)

    raw_data = raw_data[raw_data['age'] > 18]
    raw_data = raw_data[raw_data['race'].isin([' Black', ' White'])]
    print(raw_data['income'].unique())

    raw_data['race'] = raw_data['race'].replace({" Black": "Black", " White": "White"})
    raw_data['sex'] = raw_data['sex'].replace({" Female": "Female", " Male": "Male"})
    raw_data['sex'] = raw_data['sex'].replace({" Female": "Female", " Male": "Male"})
    raw_data['income'] = raw_data['income'].replace({" - 50000.": "low", " 50000+.":"high"})

    raw_data[['capital gain', 'capital loss']] = raw_data[['capital gain', 'capital loss']].astype(float)

    bin_capital_loss(raw_data)
    bin_weeks_worked_per_year(raw_data)
    bin_capital_gain(raw_data)
    bin_age(raw_data)
    bin_wage_per_hour(raw_data)

    raw_data['education'] = raw_data.apply(lambda row: bin_education(row), axis=1)

    raw_data = raw_data[['sex', 'race', 'age', 'wage per hour',  'capital gain', 'capital loss', 'weeks worked in year', 'education', 'income']]
    raw_data.to_csv('preprocessed_data/preprocessed_census.csv')
