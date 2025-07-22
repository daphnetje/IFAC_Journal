from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import inspect
from Dataset import Dataset


class BlackBoxClassifier:

    def __init__(self, classifier_name, random_state=4):
        self.classifier_name = classifier_name
        self.CLASSIFIER_MAPPING = {
        'Decision Tree': DecisionTreeClassifier,
        'Random Forest': RandomForestClassifier,
        'SVM': SVC,
        'XGB': XGBClassifier,
        'NN': MLPClassifier}
        self.random_state=random_state


    def get_classifier(self, **kwargs):
        if self.classifier_name not in self.CLASSIFIER_MAPPING:
            raise ValueError(
                f"Unsupported classifier type: {self.classifier_name}. Supported types are: {list(self.CLASSIFIER_MAPPING.keys())}")

        classifier_cls = self.CLASSIFIER_MAPPING[self.classifier_name]

        # Get the list of parameters the classifier supports
        classifier_params = inspect.signature(classifier_cls).parameters

        # If 'random_state' is a valid parameter, add it to kwargs
        if 'random_state' in classifier_params:
            kwargs['random_state'] = self.random_state

        return classifier_cls(**kwargs)

    # def fit(self, X_train_dataset, **kwargs):
    #     self.classifier = self.get_classifier(**kwargs)
    #     y_train = X_train_dataset.descriptive_data[X_train_dataset.decision_attribute]
    #     X_train = X_train_dataset.one_hot_encoded_data.loc[:, X_train_dataset.one_hot_encoded_data.columns != X_train_dataset.decision_attribute]
    #     self.classifier.fit(X_train, y_train)
    #     return self.classifier
    #
    # def fit(self, X, y):
    #     self.classifier.fit(X, y)
    #     return self.classifier

    def fit(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Dataset):
            X_train_dataset = args[0]
            y_mapping = {X_train_dataset.desirable_label: 1, X_train_dataset.undesirable_label: 0}

            self.classifier = self.get_classifier(**kwargs)
            y_train = X_train_dataset.descriptive_data[X_train_dataset.decision_attribute]
            y_train_mapped = y_train.replace(y_mapping)

            X_train = X_train_dataset.one_hot_encoded_data.loc[:,
                      X_train_dataset.one_hot_encoded_data.columns != X_train_dataset.decision_attribute]
            self.classifier.fit(X_train, y_train_mapped)
            return self.classifier
        elif len(args) == 2:
            X = args[0]
            y = args[1]
            self.classifier = self.get_classifier(**kwargs)
            self.classifier.fit(X, y)
            return self.classifier



    def predict(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Dataset):
            X_test_dataset = args[0]
            y_backwards_mapping = {1: X_test_dataset.desirable_label, 0:X_test_dataset.undesirable_label}


            y_test = X_test_dataset.descriptive_data[X_test_dataset.decision_attribute]
            X_test = X_test_dataset.one_hot_encoded_data.loc[:,
                      X_test_dataset.one_hot_encoded_data.columns != X_test_dataset.decision_attribute]

            predictions = pd.Series(self.classifier.predict(X_test))
            predictions_mapped = predictions.replace(y_backwards_mapping)
            return predictions_mapped

        else:
            single_instance = args[0]
            prediction = self.classifier.predict(single_instance)
            return prediction


    def predict_with_proba(self, X_dataset):
        y_backwards_mapping = {1: X_dataset.desirable_label, 0: X_dataset.undesirable_label}

        X = X_dataset.one_hot_encoded_data.loc[:,
                 X_dataset.one_hot_encoded_data.columns != X_dataset.decision_attribute]
        predicted_labels = pd.Series(self.classifier.predict(X))
        predictions_mapped = predicted_labels.replace(y_backwards_mapping)

        # Predict the probabilities for each class
        predicted_probabilities = self.classifier.predict_proba(X)

        # Get the probability corresponding to the predicted label for each instance
        probabilities_for_labels = pd.Series(predicted_probabilities.max(axis=1))

        return predictions_mapped, probabilities_for_labels

    #TODO doe dit mooier
    def predict_proba(self, X):
        # Predict the probabilities for each class
        predicted_probabilities = self.classifier.predict_proba(X)
        return predicted_probabilities



