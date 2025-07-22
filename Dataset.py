from sklearn.model_selection import train_test_split
import random
import pandas as pd
from copy import deepcopy
import numpy as np
np.random.seed(4)

class Dataset:
    def __init__(self, descriptive_data, ordinal_to_numeric_dicts, decision_attribute, sensitive_attributes,  reference_group_dict, undesirable_label, desirable_label, categorical_features, distance_function, one_hot_encoded_data = None):
        self.descriptive_data = descriptive_data
        self.ordinal_to_numeric_dicts = ordinal_to_numeric_dicts
        self.decision_attribute = decision_attribute
        self.sensitive_attributes = sensitive_attributes
        self.reference_group_dict = reference_group_dict
        self.undesirable_label = undesirable_label
        self.desirable_label = desirable_label
        self.categorical_features = categorical_features
        self.distance_function = distance_function
        self.binary_labels = self.decision_attribute_to_binary_array()
        self.predictions = None
        self.prediction_probabilities = None

        if one_hot_encoded_data is None:
            self.one_hot_encoded_data = self.one_hot_encode_data()
        else:
            self.one_hot_encoded_data = one_hot_encoded_data


    def set_predictions(self, predictions):
        self.predictions = predictions

    def get_predictions(self):
        return self.predictions

    def set_prediction_probabilities(self, prediction_probabilities):
        self.prediction_probabilities = prediction_probabilities

    def get_prediction_probabilities(self):
        return self.prediction_probabilities

    def __str__(self):
        return (self.descriptive_data.head(10).to_string())


    def decision_attribute_to_binary_array(self):
        decision_labels = self.descriptive_data[self.decision_attribute]
        binary_decision_labels = []
        for label in decision_labels:
            if label == self.desirable_label:
                binary_decision_labels.append(1)
            else:
                binary_decision_labels.append(0)
        return pd.Series(binary_decision_labels)

    def one_hot_encode_data(self):
        numerical_data = deepcopy(self.descriptive_data)
        for column, conversion_dict in self.ordinal_to_numeric_dicts.items():
            numerical_data[column] = numerical_data[column].replace(conversion_dict)

        df_encoded = pd.get_dummies(numerical_data, columns=self.categorical_features)
        return df_encoded


    def split_into_multiple_test_sets(self, number_of_test_sets):
        list_of_test_sets = []
        size_of_each_set = len(self.descriptive_data) // number_of_test_sets

        for i in range(number_of_test_sets):
            #sample from original dataset
            #random=4+i -> Ensures that each sample is different, but samples are consistent accross runs
            desc_data_test = self.descriptive_data.sample(n=size_of_each_set, random=4+i)
            one_hot_data_test = self.one_hot_encoded_data.loc[desc_data_test.index]

            desc_data_test = desc_data_test.reset_index(drop=True)
            one_hot_data_test = one_hot_data_test.reset_index(drop=True)

            dataset_test = Dataset(desc_data_test, self.ordinal_to_numeric_dicts, self.decision_attribute, self.undesirable_label,
                                   self.desirable_label, self.categorical_features, self.distance_function,
                                   one_hot_encoded_data=one_hot_data_test)
            list_of_test_sets.append(dataset_test)

        return list_of_test_sets


    def split_into_train_and_multiple_test_sets(self, train_size, number_of_test_sets, size_of_each_test_set=None):
        test_size = (1 - train_size)/2
        train, sit_test = self.split_into_train_test(test_size)

        train, test = train.split_into_train_test(test_size)
        list_of_test_sets = []
        if size_of_each_test_set == None:
            size_of_each_test_set = 1 / number_of_test_sets

        for i in range(number_of_test_sets):
            #sample from original dataset
            if size_of_each_test_set < 1:
                desc_data_test = test.descriptive_data.sample(frac=size_of_each_test_set, replace=True)
            else:
                desc_data_test = test.descriptive_data.sample(n=size_of_each_test_set, replace=True)

            one_hot_data_test = test.one_hot_encoded_data.loc[desc_data_test.index]

            desc_data_test = desc_data_test.reset_index(drop=True)
            one_hot_data_test = one_hot_data_test.reset_index(drop=True)

            dataset_test = Dataset(desc_data_test, self.ordinal_to_numeric_dicts, self.decision_attribute, self.sensitive_attributes, self.reference_group_dict,
                                   self.undesirable_label,self.desirable_label, self.categorical_features, self.distance_function,
                                   one_hot_encoded_data=one_hot_data_test)
            list_of_test_sets.append(dataset_test)

        return train, sit_test, list_of_test_sets



    def split_into_train_test(self, test_fraction):
        desc_data_train, desc_data_test, one_hot_data_train, one_hot_data_test = train_test_split(self.descriptive_data, self.one_hot_encoded_data, test_size=test_fraction, random_state=4)

        desc_data_train = desc_data_train.reset_index(drop=True)
        desc_data_test = desc_data_test.reset_index(drop=True)
        one_hot_data_train = one_hot_data_train.reset_index(drop=True)
        one_hot_data_test = one_hot_data_test.reset_index(drop=True)

        dataset_train = Dataset(desc_data_train, self.ordinal_to_numeric_dicts, self.decision_attribute, self.sensitive_attributes, self.reference_group_dict,
                                self.undesirable_label, self.desirable_label, self.categorical_features, self.distance_function,
                                one_hot_encoded_data=one_hot_data_train)
        dataset_test = Dataset(desc_data_test, self.ordinal_to_numeric_dicts, self.decision_attribute, self.sensitive_attributes, self.reference_group_dict,
                               self.undesirable_label, self.desirable_label, self.categorical_features, self.distance_function,
                               one_hot_encoded_data=one_hot_data_test)

        return dataset_train, dataset_test


    def give_class_label_info(self):
        instances_with_neg_label = self.descriptive_data[self.descriptive_data[self.decision_attribute] == self.undesirable_label]

        number_instances_with_neg_label = len(instances_with_neg_label)

        print("Ratio of instances with negative label: {0:.2f}".format(number_instances_with_neg_label/len(self.descriptive_data)))



    def extract_class_label_info_for_fraction_of_data(self, extract_dict):
        relevant_data = self.descriptive_data
        for key, value in extract_dict.items():
            relevant_data = relevant_data[relevant_data[key] == value]
        print(extract_dict)
        print("Number of instances: " + str(len(relevant_data)))

        instances_with_neg_label = relevant_data[relevant_data[self.decision_attribute] == self.undesirable_label]
        number_instances_with_neg_label = len(instances_with_neg_label)

        instances_with_pos_label = relevant_data[relevant_data[self.decision_attribute] == self.desirable_label]
        number_instances_with_pos_label = len(instances_with_pos_label)

        print("Ratio of instances with negative label: {0:.2f}".format(
            number_instances_with_neg_label / len(relevant_data)))
        print("______________")

        print("Ratio of instances with negative label: {0:.2f}".format(
            number_instances_with_pos_label / len(relevant_data)))
        print("______________")


    def extract_class_label_info_for_all_except_extract_dict(self, do_not_extract_dict):
        non_relevant_data = self.descriptive_data
        for key in do_not_extract_dict.keys():
            non_relevant_data = non_relevant_data[non_relevant_data[key] == do_not_extract_dict[key]]

        index_relevance_boolean_indicators = self.descriptive_data.index.isin(non_relevant_data.index)
        relevant_data = self.descriptive_data[~index_relevance_boolean_indicators]

        instances_with_neg_label = relevant_data[relevant_data[self.decision_attribute] == self.undesirable_label]
        number_instances_with_neg_label = len(instances_with_neg_label)

        print("Ratio of instances with negative label: {0:.2f}".format(
            number_instances_with_neg_label / len(relevant_data)))


def split_into_one_hot_encoded_X_and_y(data):
    decision_attribute = data.decision_attribute

    y_train = data.descriptive_data[decision_attribute]

    X_train = data.one_hot_encoded_data.loc[:, data.one_hot_encoded_data.columns != decision_attribute]

    return X_train, y_train



