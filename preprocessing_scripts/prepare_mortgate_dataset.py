import tidyhome as th
import pandas as pd


def bin_loan_amount(data):
    bin_labels = ["Less than 10k", "10k-100k", "100k-300k", "More than 300k"]
    cut_bins = [-1, 9999, 99999, 299999, 100000000000000]
    data['loan_amount'] = pd.cut(data['loan_amount'], bins=cut_bins, labels=bin_labels)
    return data


# Function to bin the debt-to-income ratio
def bin_debt_to_income(value):
    # Handle binned ranges directly
    if value.startswith('<20%'):
        return '<20%'
    elif value.startswith('20%-<30%'):
        return '20%-<30%'
    elif value.startswith('30%-<36%'):
        return '30%-<36%'
    elif value.startswith('50%-60%'):
        return '>50%'
    elif value.startswith('>60%'):
        return '>50%'
    else:
        return "36%-50%"

def bin_income(data):
    bin_labels = ["<20k", "20k-50k", "50k-100k", "100k-200k", ">200k"]
    cut_bins = [-10000, 19, 49, 99, 199, 100000000000000]
    data['income'] = pd.cut(data['income'], bins=cut_bins, labels=bin_labels)
    return data


def prepare_mortgage_data():
    #Sample for some years only 'approved' applications, since this label is underrepresented
    #orginally
    #data_2018 = th.get_loans(2018, "dc", [th.Action.APPROVED], [th.Race.BLACK, th.Race.WHITE])

    data_2018_w_approved = th.get_loans(2018, "dc", [th.Action.APPROVED], [th.Race.WHITE])
    data_2018_b_denied = th.get_loans(2018, "dc", [th.Action.DENIED], [th.Race.BLACK]).sample(frac=0.2)

    data_2019 = th.get_loans(2019, "dc", [th.Action.APPROVED], [th.Race.BLACK, th.Race.WHITE])
    data_2020 = th.get_loans(2020, "dc", [th.Action.APPROVED, th.Action.DENIED], [th.Race.BLACK, th.Race.WHITE])
    data_2021 = th.get_loans(2021, "dc", [th.Action.APPROVED, th.Action.DENIED], [th.Race.BLACK, th.Race.WHITE])


    data = pd.concat([data_2018_w_approved, data_2018_b_denied, data_2019, data_2020, data_2021]).reset_index(drop=True)
    data = data.dropna(subset=["derived_sex", "derived_race", "income",  "loan_amount", "debt_to_income_ratio", "loan_purpose", "action_taken"]).reset_index(drop=True)
    data = data[data["derived_sex"] != "Sex Not Available"]

    action_taken_dict = {2: 'Approved', 3: 'Denied'}
    loan_purpose_dict = {1: "Home purchase", 2: "Home improvement", 31: "Refinancing", 32: "Cash-out refinancing", 4: "Other purpose", 5: "Not applicable"}
    data = data.replace({"action_taken":action_taken_dict, "loan_purpose": loan_purpose_dict})
    data = bin_loan_amount(data)

    data = bin_income(data)

    data = data[data['debt_to_income_ratio'] != 'Exempt']
    data = data.reset_index(drop=True)
    data['debt_to_income_ratio'] = data['debt_to_income_ratio'].apply(bin_debt_to_income)

    data = data[["derived_sex", "derived_race", "income",  "loan_amount", "debt_to_income_ratio", "loan_purpose", "action_taken"]]
    data.to_excel("preprocessed_data/mortgage.xlsx")