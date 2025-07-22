import pandas as pd

# writing a short function to merge data on keys except that 'inner'
def combine(df1,df2,key):
    return df1.merge(df2, how = 'inner',suffixes=('', '_y'),left_on = key, right_on = key)


def prepare_open_university_data():
    assessments_info = pd.read_csv("LearningAnalyticsData/assessments.csv")
    assessments_scores_students = pd.read_csv("LearningAnalyticsData/studentAssessment.csv")
    student_info = pd.read_csv("studentInfo.csv")

    assessment_infos_and_scores = combine(assessments_scores_students, assessments_info, 'id_assessment')  # dataframes

    studentPerformance = combine(student_info, assessment_infos_and_scores, 'id_student')

    studentPerformance['SES'] = studentPerformance['imd_band']
    studentPerformance['SES'].unique()
    studentPerformance.loc[(studentPerformance['SES'] == '90-100%') | (studentPerformance['SES'] == '80-90%') | (
                studentPerformance['SES'] == '70-80%') | (studentPerformance['SES'] == '60-70%') | (
                studentPerformance['SES'] == '50-60%'), 'SES'] = 'high'

    studentPerformance.loc[(studentPerformance['SES'] == '0-10%') | (studentPerformance['SES'] == '10-20') |
                           (studentPerformance['SES'] == '20-30%') | (studentPerformance['SES'] == '30-40%') |
                           (studentPerformance['SES'] == '40-50%'), 'SES'] = 'low'

    studentMeanScore = studentPerformance.groupby(
        ['id_student', 'gender', 'region', 'highest_education', 'imd_band', 'SES', 'age_band', 'disability',
         'num_of_prev_attempts', 'studied_credits',
         'final_result', 'code_module', 'code_presentation', 'assessment_type'], as_index=False)['score'].mean()


    sorted = studentMeanScore.sort_values(by=['id_student', 'assessment_type'])

    students_with_both = sorted.groupby('id_student')['final_result'].apply(lambda x: {'Pass', 'Fail', 'Withdrawn'}.issubset(set(x)))
    students_with_both_ids = students_with_both[students_with_both].index

    # View the rows of these students
    students_with_both_df = sorted[sorted['id_student'].isin(students_with_both_ids)]

    print("Students with both 'pass' and 'fail':")

    relevant_data = studentMeanScore[['code_presentation','num_of_prev_attempts','studied_credits',
                                       'gender', 'region', 'highest_education','SES', 'age_band',
                                       'disability','assessment_type', 'score', 'final_result']]
    relevant_data = relevant_data.dropna()
    print("...")


def bin_studied_credits(data):
    cut_labels_credits = ["0-59", "60-65", "More than 65"]
    cut_bins_credits = [-1, 59, 65, 1000]
    data['studied_credits'] = pd.cut(data['studied_credits'], bins=cut_bins_credits,
                                     labels=cut_labels_credits)
    return data

def prepare_simple_open_university_data():
    student_info = pd.read_csv("studentInfo.csv")

    student_info = student_info.dropna()
    student_info = student_info[student_info['final_result'] != 'Withdrawn']
    student_info.loc[student_info['final_result'] == 'Distinction', 'final_result'] = "Pass"

    student_info['SES'] = student_info['imd_band']
    student_info.loc[(student_info['SES'] == '90-100%') | (student_info['SES'] == '80-90%') | (
                student_info['SES'] == '70-80%') | (student_info['SES'] == '60-70%') | (
                student_info['SES'] == '50-60%'), 'SES'] = 'high'

    student_info.loc[(student_info['SES'] == '0-10%') | (student_info['SES'] == '10-20') |
                           (student_info['SES'] == '20-30%') | (student_info['SES'] == '30-40%') |
                           (student_info['SES'] == '40-50%'), 'SES'] = 'low'


    student_info = bin_studied_credits(student_info)

    student_info = student_info[["gender", "disability", "age_band", "SES", "highest_education", "studied_credits", "code_module", "final_result"]]
    student_info.to_csv("OULAD.csv")
    return
