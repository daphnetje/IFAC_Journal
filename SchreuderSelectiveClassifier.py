from dpabst.post_process import TransformDPAbstantion
from IFAC.Reject import SchreuderReject, Reject, SchreuderFlip
from IFAC.BlackBoxClassifier import BlackBoxClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SelectiveClassifierSchreuder:

    def __init__(self, reject_rate_per_sens_group, base_classifier):
        self.reject_rate_per_sens_group = reject_rate_per_sens_group
        self.base_classifier = base_classifier

    def is_reference(self, row):
        for ref_group in self.reference_groups:
            if (pd.Series(ref_group).equals(row[self.sensitive_attributes])):
                return True
        return False

    def adapt_data_to_right_format(self, dataset):
        X = dataset.one_hot_encoded_data.loc[:, dataset.one_hot_encoded_data.columns != dataset.decision_attribute]
        is_reference_group = dataset.descriptive_data.apply(self.is_reference, axis=1).astype(int)
        X['is_favoured'] = is_reference_group
        for sens_att in self.sensitive_attributes:
            sensitive_columns_one_hot = [col for col in X if col.startswith(sens_att)]
            X = X.drop(columns = sensitive_columns_one_hot)

        #sensitive_columns_one_hot = [col for col in X if col.startswith(self.sensitive_attribute)]
        # sens_info_to_keep = X[sensitive_columns[0]]
        #X = X.drop(columns = sensitive_columns_one_hot)
        # X[self.sensitive_attribute]= sens_info_to_keep

        y = dataset.descriptive_data[dataset.decision_attribute]
        y_numeric = y.replace({dataset.desirable_label:1, dataset.undesirable_label:0})
        return X.values, y_numeric.values

    def transform_predictions_to_reject_format(self, dataset_test, X_test, predictions):
        descriptive_test = dataset_test.descriptive_data
        right_format = []
        flips = []
        rejects = []
        index_counter = 0
        n_rejected = 0
        n_flipped = 0
        for prediction in predictions:
            formatted_instance = X_test[index_counter].reshape(1, -1)
            descriptive_instance = descriptive_test.loc[index_counter].to_dict()
            org_prediction = self.trained_bb.predict(formatted_instance)[0]
            org_prediction_label = dataset_test.desirable_label if org_prediction == 1 else dataset_test.undesirable_label
            org_prediction_probability = self.trained_bb.predict_proba(formatted_instance)[0][org_prediction]

            if (prediction == 1) and (org_prediction == 1):
                right_format.append(dataset_test.desirable_label)
            elif (prediction == 0) and (org_prediction == 0):
                right_format.append(dataset_test.undesirable_label)
            elif (prediction == 0) and (org_prediction == 1):
                right_format.append(dataset_test.undesirable_label)
                flips.append(SchreuderFlip(descriptive_instance, org_prediction_label, org_prediction_probability))
                n_flipped += 1
            elif (prediction == 1) and (org_prediction == 0):
                right_format.append(dataset_test.desirable_label)
                flips.append(SchreuderFlip(descriptive_instance, org_prediction_label, org_prediction_probability))
                n_flipped += 1
            else:
                n_rejected += 1
                reject = SchreuderReject(descriptive_instance, org_prediction_label, org_prediction_probability)
                rejects.append(reject)
                right_format.append(reject)
            index_counter += 1
        print("Schreuder rejected: " + str(n_rejected) + " and flipped: " + str(n_flipped))
        predictions = pd.Series(right_format)
        flips = pd.Series(flips)
        rejects = pd.Series(rejects)
        return predictions, flips, rejects


    def fit(self, train):
        self.sensitive_attributes = train.sensitive_attributes
        self.reference_groups = [train.reference_group_dict]

        X_train, y_train = self.adapt_data_to_right_format(train)
        #fit base classifier
        # Step 1: Train Black-Box Model
        self.BB = BlackBoxClassifier(self.base_classifier)
        self.trained_bb = self.BB.fit(X_train, y_train)

        self.transformer = TransformDPAbstantion(self.trained_bb, self.reject_rate_per_sens_group)
        self.transformer.fit(X_train)

    def predict(self, test):
        formatted_X_test, formatted_y_test = self.adapt_data_to_right_format(test)

        y_pred = self.transformer.predict(formatted_X_test)
        y_pred_reject_format, flips, rejects = self.transform_predictions_to_reject_format(test, formatted_X_test, y_pred)
        return y_pred_reject_format, flips, rejects

    def compare_and_predict(self, test):
        X_test, y_test = self.adapt_data_to_right_format(test)

        regular_bb_predictions = self.BB.predict(X_test)
        regular_bb_predictions_probas = self.BB.predict_proba(X_test)
        predictions_with_reject = self.transformer.predict(X_test)
        y_pred_reject_format = self.transform_predictions_to_reject_format(test, X_test, predictions_with_reject)

        compare_bb_and_schreuder(regular_bb_predictions, regular_bb_predictions_probas, y_pred_reject_format, test.descriptive_data)

def compare_bb_and_schreuder(regular_bb_predictions, regular_bb_predictions_probas, y_pred_reject_format, test_data):
    index = 0

    number_of_pos_flipped_to_neg = 0
    probas_of_pos_flipped_to_neg = []
    instances_with_pos_flipped_to_neg = []

    number_of_neg_flipped_to_pos = 0
    probas_of_neg_flipped_to_pos = []
    instances_with_neg_flipped_to_pos = []

    number_of_neg_rejected = 0
    probas_of_neg_rejected = []
    instances_with_neg_rejected = []

    number_of_pos_rejected = 0
    probas_of_pos_rejected = []
    instances_with_pos_rejected = []

    probas_of_pos_men_kept = []
    probas_of_neg_women_kept = []

    for pred in regular_bb_predictions:
        pred_proba = max(regular_bb_predictions_probas[index])
        schreuder_pred = y_pred_reject_format[index]
        test_instance = test_data.loc[index]

        if isinstance(schreuder_pred, Reject) and (pred == 1):
            number_of_pos_rejected += 1
            probas_of_pos_rejected.append(pred_proba)
            instances_with_pos_rejected.append(test_instance)

        elif isinstance(schreuder_pred, Reject) and (pred == 0):
            number_of_neg_rejected += 1
            probas_of_neg_rejected.append(pred_proba)
            instances_with_neg_rejected.append(test_instance)

        elif (schreuder_pred == 'Fail') and (pred == 1):
            number_of_pos_flipped_to_neg += 1
            probas_of_pos_flipped_to_neg.append(pred_proba)
            instances_with_pos_flipped_to_neg.append(test_instance)

        elif (schreuder_pred == 'Pass') and (pred == 0):
            number_of_neg_flipped_to_pos += 1
            probas_of_neg_flipped_to_pos.append(pred_proba)
            instances_with_neg_flipped_to_pos.append(test_instance)

        elif (schreuder_pred == 'Pass') and (pred == 1) and (test_instance['disability'] == 'N'):
            probas_of_pos_men_kept.append(pred_proba)

        elif (schreuder_pred == 'Fail') and (pred == 0) and (test_instance['disability'] == 'Y'):
            probas_of_neg_women_kept.append(pred_proba)

        index += 1

    print("Number of rejected pos preds", number_of_pos_rejected)
    print("Number of rejected neg preds", number_of_neg_rejected)
    print("Number of flipped pos preds", number_of_pos_flipped_to_neg)
    print("Number of flipped neg preds", number_of_neg_flipped_to_pos)

    arrays = [probas_of_neg_rejected, probas_of_neg_flipped_to_pos,  probas_of_neg_women_kept, probas_of_pos_rejected, probas_of_pos_flipped_to_neg, probas_of_pos_men_kept]
    colors = ['red', 'blue', 'orange', 'green', 'yellow', 'black']

    plt.figure(figsize=(8, 6))

    for i, (arr, color) in enumerate(zip(arrays, colors)):
        # Add random jitter on x-axis to avoid overlap
        x = np.random.normal(loc=i, scale=0.05, size=len(arr))
        y = arr
        plt.scatter(x, y, color=color, label=f'Array {i + 1}', alpha=0.6, edgecolor='k')

    # Optional: better x-ticks
    plt.xticks(range(len(arrays)), [f'Array {i + 1}' for i in range(len(arrays))])
    plt.ylabel('Value')
    plt.title('Scatter Plot of Float Arrays')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return






    # def learn_and_apply_reject_classification(self, dataset_train, dataset_test):
    #     X_train, y_train = self.adapt_data_to_right_format(dataset_train)
    #     X_test, y_test = self.adapt_data_to_right_format(dataset_test)
    #
    #     bb_classifier = self.train_base_classifier(X_train, y_train)
    #     transformer = TransformDPAbstantion(bb_classifier, self.reject_rate_per_sens_group)
    #     transformer.fit(X_test)
    #     y_pred = transformer.predict(X_test)
    #     y_pred_reject_format = self.transform_predictions_to_reject_format(bb_classifier, dataset_test, X_test, y_pred, dataset_train.desirable_label, dataset_train.undesirable_label)
    #     print(y_pred_reject_format)
    #     return y_pred_reject_format
