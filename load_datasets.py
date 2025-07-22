from Dataset import Dataset
import pandas as pd

def load_income_data():
    raw_data = pd.read_csv('preprocessed_data/income_sample.csv')
    raw_data = raw_data.sample(n=16000, random_state=4)   #16000

    print("Size complete data: ", len(raw_data))
    descriptive_dataframe = raw_data[
        ['age', 'marital status', 'education', 'workinghours', 'workclass', 'occupation', 'race', 'sex', 'income']]

    age_dict = {"Younger than 25": 1, "25-29": 2, "30-39": 3, "40-49": 4, "50-59": 5, "60-69": 6, "Older than 70": 7}
    education_dict = {"No Elementary School": 1, "Elementary School": 2, "Middle School": 3,
                      "Started High School, No Diploma": 4, "High School or GED Diploma": 5,
                      "Started College, No Diploma": 6, "Associate Degree": 7, "Bachelor Degree": 8,
                      "Master or other Degree Beyond Bachelor": 9, "Doctorate Degree": 10}
    workinghours_dict = {"Less than 20": 1, "20-39": 2, "40-49": 3, "More than 50": 4}
    dicts_ordinal_to_numeric = {'age': age_dict, 'education': education_dict, 'workinghours': workinghours_dict}

    categorical_features = ['marital status', 'occupation', 'workclass', 'race', 'sex']
    dataset = Dataset(descriptive_dataframe, dicts_ordinal_to_numeric, decision_attribute="income", sensitive_attributes= ["sex", "race"],
                      reference_group_dict={'sex': 'Male', 'race': 'White alone'}, undesirable_label="low",
                      desirable_label="high", categorical_features=categorical_features,
                      distance_function=distance_function_income_pred)

    return dataset

#order of features: ['age_num', 'marital status', 'education_num', 'workinghours_num', 'workclass', 'occupation', 'race', 'sex', 'income']]
def distance_function_income_pred(x1, x2):
    age_dict = {"Younger than 25": 1, "25-29": 2, "30-39": 3, "40-49": 4, "50-59": 5, "60-69": 6, "Older than 70": 7}
    age_diff = abs(age_dict[x1[0]] - age_dict[x2[0]]) / 6

    if x1[1] == x2[1]:
        marital_status_diff = 0
    else:
        marital_status_diff = 0.5

    education_dict = {"No Elementary School":1, "Elementary School":2, "Middle School":3,
                            "Started High School, No Diploma":4, "High School or GED Diploma":5,
                            "Started College, No Diploma":6, "Associate Degree":7, "Bachelor Degree":8,
                            "Master or other Degree Beyond Bachelor":9, "Doctorate Degree":10}
    education_diff = abs(education_dict[x1[2]] - education_dict[x2[2]]) / 9

    workinghours_dict = {"Less than 20": 1, "20-39": 2, "40-49": 3, "More than 50": 4}
    workinghours_diff = abs(workinghours_dict[x1[3]] - workinghours_dict[x2[3]])/3

    if x1[4] == x2[4]:
        workclass_diff = 0
    else:
        workclass_diff = 0.5

    if x1[5] == x2[5]:
        occupation_diff = 0
    else:
        occupation_diff = 0.5

    return age_diff + marital_status_diff + education_diff + workinghours_diff + workclass_diff + occupation_diff


def load_grade_prediction_data():
    descriptive_dataframe = pd.read_csv('preprocessed_data/processed_student_alc.csv')
    # Remove columns with 'Unnamed:' in their name
    descriptive_dataframe = descriptive_dataframe.loc[:, ~descriptive_dataframe.columns.str.contains('^Unnamed')]

    studytime_dict = {'less than 2 hours':1, '2-5 hours':2, 'more than 5 hours':3}
    freetime_dict = {'low':1, 'average':2, 'high':3}
    alcohol_consumption_dict = {'low':1, 'moderate':2, 'high':3, 'very high':4}
    go_out_dict = {'never':1, 'once a week':2, 'twice a week':3, 'three times or more':4}
    parents_edu_dict = {'no education':1, 'middle school':2, 'high school':3, 'university':4}
    absences_dict = {'0-1':1, '2-6':2, 'More than 7':3}
    past_performances_dict = {'Pass': 1, 'Fail': 0}


    dicts_ordinal_to_numeric = {'studytime':studytime_dict, 'freetime': freetime_dict, 'Walc': alcohol_consumption_dict, \
                                'goout':go_out_dict, 'Parents_edu': parents_edu_dict, 'absences': absences_dict, \
                                'G3': past_performances_dict}

    categorical_features = ['reason', 'sex']

    dataset = Dataset(descriptive_dataframe, dicts_ordinal_to_numeric, decision_attribute="Predicted Pass",
                      undesirable_label="fail",
                      desirable_label="pass", categorical_features=categorical_features,
                      distance_function=distance_function_grade_pred)
    return dataset


#studytime,freetime,Walc,goout,Parents_edu,absences,reason,G3,sex,Predicted Pass,GroundTruth
def distance_function_grade_pred(x1, x2):
    studytime_dict = {'less than 2 hours': 1, '2-5 hours': 2, 'more than 5 hours': 3}
    studytime_diff = abs(studytime_dict[x1[0]] - studytime_dict[x1[0]])/2

    freetime_dict = {'low': 1, 'average': 2, 'high': 3}
    freetime_diff = abs(freetime_dict[x1[1]] - freetime_dict[x1[1]]) / 2

    walc_dict = {'low': 1, 'moderate': 2, 'high': 3, 'very high': 4}
    walc_diff = abs(walc_dict[x1[2]] - walc_dict[x2[2]]) / 3

    go_out_dict = {'never': 1, 'once a week': 2, 'twice a week': 3, 'three times or more': 4}
    goout_diff = abs(go_out_dict[x1[3]] - go_out_dict[x2[3]]) / 3

    parents_edu_dict = {'no education':1, 'middle school':2, 'high school':3, 'university':4}
    parents_edu_diff = abs(parents_edu_dict[x1[4]] - parents_edu_dict[x2[4]]) / 3

    absences_dict = {'0-1': 1, '2-6': 2, 'More than 7': 3}
    absences_diff = abs(absences_dict[x1[5]] - absences_dict[x2[5]])/2

    if x1[6] == x2[6]:
        reason_diff = 0
    else:
        reason_diff = 0.1

    g3_dict = {'Pass': 1, 'Fail': 0}
    g3_diff = abs(g3_dict[x1[7]] - g3_dict[x2[7]])

    return studytime_diff + freetime_diff + walc_diff + goout_diff + parents_edu_diff + absences_diff + reason_diff + g3_diff


def load_OULAD():
    descriptive_dataframe = pd.read_csv('preprocessed_data/OULAD.csv')
    # Remove columns with 'Unnamed:' in their name
    descriptive_dataframe = descriptive_dataframe.loc[:, ~descriptive_dataframe.columns.str.contains('^Unnamed')]

    qualification_dict = {"No Formal quals": 1, "Lower Than A Level": 2, "A Level or Equivalent": 3 , "HE Qualification": 4, "Post Graduate Qualification": 5}
    age_dict = {"0-35": 1, "35-55": 2, "55<=": 3}
    credits_dicts = {"0-59": 1, "60-65": 2, "More than 65": 3}
    ses_dict = {"low": 1, 'high': 2}


    dicts_ordinal_to_numeric = {'highest_education':qualification_dict, 'studied_credits': credits_dicts, 'age_band': age_dict, "SES": ses_dict}

    categorical_features = ['gender','disability','code_module']

    dataset = Dataset(descriptive_dataframe, dicts_ordinal_to_numeric, decision_attribute="final_result",
                      sensitive_attributes=["disability"], reference_group_dict= {'disability': 'N'}, undesirable_label="Fail",
                      desirable_label="Pass", categorical_features=categorical_features,
                      distance_function=distance_function_oulad)
    return dataset


#'gender', 'disability', 'age_band', 'SES', 'highest_education', 'studied_credits', 'code_module', 'final_result'
def distance_function_oulad(x1, x2):
    age_dict = {"0-35": 1, "35-55": 2, "55<=": 3}
    age_diff = abs(age_dict[x1[2]] - age_dict[x2[2]])/2

    edu_dict = {"No Formal quals": 1, "Lower Than A Level": 2, "A Level or Equivalent": 3 , "HE Qualification": 4, "Post Graduate Qualification": 5}
    edu_diff = abs(edu_dict[x1[4]] - edu_dict[x2[4]])/4

    credits_dicts = {"0-59": 1, "60-65": 2, "More than 65": 3}
    credits_diff = abs(credits_dicts[x1[5]] - credits_dicts[x2[5]])/2

    if x1[6] == x2[6]:
        module_diff = 0
    else:
        module_diff = 0.5

    return age_diff+edu_diff+credits_diff+module_diff


def load_mortgage():
    descriptive_dataframe = pd.read_excel("preprocessed_data/mortgage.xlsx")
    # Remove columns with 'Unnamed:' in their name
    descriptive_dataframe = descriptive_dataframe.loc[:, ~descriptive_dataframe.columns.str.contains('^Unnamed')]

    loan_amount_dict = {"Less than 10k": 1, "10k-100k": 2, "100k-300k": 3, "More than 300k": 4}
    income_dict = {"<20k": 1, "20k-50k": 2, "50k-100k": 3, "100k-200k": 4, ">200k": 5}
    debt_to_income_dict = {"<20%" : 1, "20%-<30%": 2, "30%-<36%": 3, "36%-50%": 4, ">50%": 5}


    dicts_ordinal_to_numeric = {'loan_amount': loan_amount_dict, 'income': income_dict,
                                'debt_to_income_ratio':debt_to_income_dict}

    categorical_features = ["derived_sex", "derived_race", "loan_purpose"]

    dataset = Dataset(descriptive_dataframe, dicts_ordinal_to_numeric, decision_attribute="action_taken",
                      sensitive_attributes=["derived_race"], reference_group_dict={'derived_race': 'White'},
                      undesirable_label="Denied",
                      desirable_label="Approved", categorical_features=categorical_features,
                      distance_function=distance_function_mortgage)
    return dataset

#"derived_sex", "derived_race", "income",  "loan_amount", "debt_to_income_ratio", "loan_purpose", "action_taken"
def distance_function_mortgage(x1, x2):
    income_dict = {"<20k": 1, "20k-50k": 2, "50k-100k": 3, "100k-200k": 4, ">200k": 5}
    income_diff = abs(income_dict[x1[2]] - income_dict[x2[2]])/4

    loan_amount_dict = {"Less than 10k": 1, "10k-100k": 2, "100k-300k": 3, "More than 300k": 4}
    loan_amount_diff = abs(loan_amount_dict[x1[3]] - loan_amount_dict[x2[3]]) / 3

    debt_to_income_dicts = {"<20%" : 1, "20%-<30%": 2, "30%-<36%": 3, "36%-50%": 4, ">50%": 5}
    debt_to_income_diff = abs(debt_to_income_dicts[x1[4]] - debt_to_income_dicts[x2[4]]) / 4

    if x1[5] == x2[5]:
        purpose_diff = 0
    else:
        purpose_diff = 0.5

    return income_diff + loan_amount_diff + debt_to_income_diff + purpose_diff



def load_recidivism():
    descriptive_dataframe = pd.read_csv('preprocessed_data/recidivism_data.csv', keep_default_na=False,na_values=['NaN'])
    # Remove columns with 'Unnamed:' in their name
    descriptive_dataframe = descriptive_dataframe.loc[:, ~descriptive_dataframe.columns.str.contains('^Unnamed')]
    descriptive_dataframe = descriptive_dataframe.sample(n=20000, random_state=4)


    prior_criminal_count_dict = {"None": 0, "1-5": 1, "6-10": 2, "More than 10": 3}
    age_dict = {"Younger than 18": 1, "18-29": 2, "30-39": 3, "40-49": 4, "50-59": 5, "Older than 60": 6}

    dicts_ordinal_to_numeric = {'prior_felony': prior_criminal_count_dict, 'prior_misdemeanor': prior_criminal_count_dict, \
                                'prior_criminal_traffic': prior_criminal_count_dict, 'age': age_dict}

    categorical_features = ['case_type', 'offense', 'race']

    dataset = Dataset(descriptive_dataframe, dicts_ordinal_to_numeric, decision_attribute="recidivism",
                      undesirable_label="yes", sensitive_attributes=['race'], reference_group_dict={'race': 'Caucasian'},
                      desirable_label="no", categorical_features=categorical_features,
                      distance_function=distance_function_recidivism)
    return dataset

#race,age,case_type,offense,prior_felony,prior_misdemeanor,prior_criminal_traffic,recidivism
def distance_function_recidivism(x1, x2):
    age_dict = {"Younger than 18": 1, "18-29": 2, "30-39": 3, "40-49": 4, "50-59": 5, "Older than 60": 6}
    age_diff = (abs(age_dict[x1[1]] - age_dict[x2[1]])) / 5

    if x1[2] == x2[2]:
        case_type_diff = 0
    else:
        case_type_diff = 1

    if x1[3] == x2[3]:
        offense_diff = 0
    else:
        offense_diff = 0.5

    prior_criminal_count_dict = {"None": 0, "1-5": 1, "6-10": 2, "More than 10": 3}
    prior_felony_diff = (abs(prior_criminal_count_dict[x1[4]] - prior_criminal_count_dict[x2[4]])) / 3
    prior_misdemeanor_diff = (abs(prior_criminal_count_dict[x1[5]] - prior_criminal_count_dict[x2[5]])) / 3
    prior_criminal_traffic_diff = (abs(prior_criminal_count_dict[x1[6]] - prior_criminal_count_dict[x2[6]])) / 3

    return age_diff + case_type_diff + offense_diff + prior_felony_diff + prior_misdemeanor_diff + prior_criminal_traffic_diff


#Credit Amount,Duration of Credit (month),Age (years),Account Balance,Occupation,Length of current employment,Payment Status of Previous Credit
def load_german_credit():
    descriptive_dataframe = pd.read_csv('preprocessed_data/german_preprocessed.csv')
    descriptive_dataframe = descriptive_dataframe.loc[:, ~descriptive_dataframe.columns.str.contains('^Unnamed')]
    descriptive_dataframe = descriptive_dataframe[['Payment Status of Previous Credit', 'Account Balance', 'Credit Amount', 'Occupation', 'Sex', 'Creditability']]
    credit_amount_dict = {"Less than 1000": 0, "1000-2999": 1, "3000-6999": 2, "More than 7000": 3}
    credit_duration_dict = {"Less than a year": 0, "1-2 years": 1, "2-3 years": 2, "More than 3 years": 3}
    age_dict = {"Younger than 30": 1, "Older than 30": 2}
    account_balance_dict = {"No Account": 0, "None": 1, "Below 200DM": 2, "200DM or above": 3}
    occupation_dict = {"Unskilled": 0, "Skilled": 1, "Highly Skilled": 2}
    length_of_employment_dict = {"Unemployed": 0, "Less than a year": 1, "1-3 years": 2, "More than 4 years": 3}
    previous_credit_dict = {"NA/Not Delayed": 0, "Delays": 1}

    dicts_ordinal_to_numeric = {'Credit Amount': credit_amount_dict,
                                'Duration of Credit (month)': credit_duration_dict, \
                                'Age (years)': age_dict, 'Account Balance': account_balance_dict, \
                                'Occupation': occupation_dict, 'Length of current employment': length_of_employment_dict, \
                                'Payment Status of Previous Credit': previous_credit_dict}

    dicts_ordinal_to_numeric = {'Account Balance': account_balance_dict,  \
                                'Payment Status of Previous Credit': previous_credit_dict, 'Credit Amount': credit_amount_dict,
                                'Occupation': occupation_dict}

    categorical_features = ['Sex']

    dataset = Dataset(descriptive_dataframe, dicts_ordinal_to_numeric, decision_attribute="Creditability",
                      undesirable_label="Not Credible",
                      desirable_label="Credible", categorical_features=categorical_features,
                      distance_function=distance_function_german_credit)
    return dataset


#'Payment Status of Previous Credit', 'Account Balance', 'Credit Amount', 'Occupation', 'Sex',
def distance_function_german_credit(x1, x2):
    previous_credit_dict = {"NA/Not Delayed": 0, "Delays": 1}
    previous_credits_diff = (abs(previous_credit_dict[x1[0]] -previous_credit_dict[x2[0]]))

    account_balance_dict = {"No Account": 0, "None": 1, "Below 200DM": 2, "200DM or above": 3}
    account_balance_diff = (abs(account_balance_dict[x1[1]] - account_balance_dict[x2[1]])) / 3

    credit_amount_dict = {"Less than 1000": 0, "1000-2999": 1, "3000-6999": 2, "More than 7000": 3}
    credit_amount_diff = (abs(credit_amount_dict[x1[2]] - credit_amount_dict[x2[2]])) / 3

    occupation_dict = {"Unskilled": 0, "Skilled": 1, "Highly Skilled": 2}
    occupation_diff = (abs(occupation_dict[x1[3]] - occupation_dict[x2[3]])) / 2


def load_census_data():
    descriptive_dataframe = pd.read_csv('preprocessed_data/preprocessed_census.csv')

    # Select a random 60% of rows where income is "high"
    high_income_df = descriptive_dataframe[descriptive_dataframe['income'] == 'high'].sample(frac=0.6, random_state=42)
    print(len(high_income_df))
    #high_income_df = high_income_df.sample(frac=0.5, random_state=42)
    # Select a random 10% of rows where income is "low"
    low_income_df = descriptive_dataframe[descriptive_dataframe['income'] == 'low'].sample(frac=0.1, random_state=42)
    print(len(low_income_df))
    # Combine both subsets into a new dataframe
    descriptive_dataframe = pd.concat([high_income_df, low_income_df])
    print(len(descriptive_dataframe))

    capital_gain_dict = {"<=500": 0, ">500": 1}
    capital_loss_dict = {"<=500": 0, ">500": 1}
    age_dict = {"Younger than 25": 0, "26-60": 1, "Older than 60": 2}
    wage_dict = {"<500": 0, "500-1000": 1, "More than 1000": 2}
    weeks_worked_dict = {">=26": 0, "27-51": 1, "=52": 2}
    education_dict = {"Elementary School": 0, "Middle School": 1, "High School, no diploma": 2, "High School Degree": 3, "College or Associate": 4, "University Degree": 5, "Professor/Doctorate": 6}

    dicts_ordinal_to_numeric = {'capital gain': capital_gain_dict, 'capital loss': capital_loss_dict,
                                'age': age_dict, 'weeks worked in year': weeks_worked_dict, 'wage per hour': wage_dict,
                                'education': education_dict}

    categorical_features = ['sex']

    dataset = Dataset(descriptive_dataframe, dicts_ordinal_to_numeric, decision_attribute="income",
                      sensitive_attributes=['sex'], reference_group_dict={'sex': 'Male'}, undesirable_label="low",
                      desirable_label="high", categorical_features=categorical_features,
                      distance_function=distance_function_census)
    return dataset


# 'sex', 'race', 'age', 'wage per hour',  'capital gain', 'capital loss', 'weeks worked in year', 'education', 'income']
def distance_function_census(x1, x2):
    age_dict = {"Younger than 25": 0, "26-60": 1, "Older than 60": 2}
    age_diff = abs(age_dict[x1[1]] - age_dict[x2[1]]) / 2

    wage_dict = {"<500": 0, "500-1000": 1, "More than 1000": 2}
    wage_diff = abs(wage_dict[x1[2]] - wage_dict[x2[2]]) / 2

    capital_gain_dict = {"<=500": 0, ">500": 1}
    capital_gain_diff = abs(capital_gain_dict[x1[3]] - capital_gain_dict[x2[3]]) / 1

    capital_loss_dict = {"<=500": 0, ">500": 1}
    capital_loss_diff = abs(capital_loss_dict[x1[4]] - capital_loss_dict[x2[4]]) / 1

    weeks_worked_dict = {">=26": 0, "27-51": 1, "=52": 2}
    weeks_worked_diff = abs(weeks_worked_dict[x1[5]] - weeks_worked_dict[x2[5]]) / 1

    education_dict = {"Elementary School": 0, "Middle School": 1, "High School, no diploma": 2, "High School Degree": 3, "College or Associate": 4, "University Degree": 5, "Professor/Doctorate": 6}
    education_diff = abs(education_dict[x1[6]] - education_dict[x2[6]]) / 6

    return age_diff + wage_diff + capital_gain_diff + capital_loss_diff + weeks_worked_diff + education_diff

# def distance_function_census(x1, x2):
#     age_dict = {"Younger than 25": 0, "26-60": 1, "Older than 60": 2}
#     age_diff = abs(age_dict[x1[2]] - age_dict[x2[2]]) / 2
#
#     wage_dict = {"<500": 0, "500-1000": 1, "More than 1000": 2}
#     wage_diff = abs(wage_dict[x1[3]] - wage_dict[x2[3]]) / 2
#
#     capital_gain_dict = {"<=500": 0, ">500": 1}
#     capital_gain_diff = abs(capital_gain_dict[x1[4]] - capital_gain_dict[x2[4]]) / 1
#
#     capital_loss_dict = {"<=500": 0, ">500": 1}
#     capital_loss_diff = abs(capital_loss_dict[x1[5]] - capital_loss_dict[x2[5]]) / 1
#
#     weeks_worked_dict = {">=26": 0, "27-51": 1, "=52": 2}
#     weeks_worked_diff = abs(weeks_worked_dict[x1[6]] - weeks_worked_dict[x2[6]]) / 1
#
#     education_dict = {"Elementary School": 0, "Middle School": 1, "High School, no diploma": 2, "High School Degree": 3, "College or Associate": 4, "University Degree": 5, "Professor/Doctorate": 6}
#     education_diff = abs(education_dict[x1[7]] - education_dict[x2[7]]) / 6
#
#     return age_diff + wage_diff + capital_gain_diff + capital_loss_diff + weeks_worked_diff + education_diff

