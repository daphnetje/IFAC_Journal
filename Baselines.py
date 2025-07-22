from IFAC.BlackBoxClassifier import BlackBoxClassifier
import pandas as pd
import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import sklearn.metrics as skm
from IFAC.Reject import create_uncertainty_based_reject


# get unique values in numpy array, pandas series or list
def unique(y):
    if isinstance(y, np.ndarray):
        return np.unique(y)
    if isinstance(y, pd.Series):
        return np.unique(y.values)
    if isinstance(y, list):
        return np.array(set(y))
    raise RuntimeError('unknown data type', type(y))

class UBAC:
    def __init__(self, coverage, val_ratio, base_classifier):
        self.coverage = coverage
        self.val_ratio = val_ratio
        self.base_classifier = base_classifier

    def fit(self, X):
        val_n = int(self.val_ratio * len(X.descriptive_data))
        X_train_dataset, X_val_dataset = X.split_into_train_test(val_n)

        n_to_reject = int((1-self.coverage) * val_n)

        # Step 1: Train Black-Box Model
        self.BB = BlackBoxClassifier(self.base_classifier)
        self.BB.fit(X_train_dataset)

        #Step 2: Apply on validation data
        pred_val, proba_val = self.BB.predict_with_proba(X_val_dataset)

        #Step 3: Learn threshold
        self.threshold = self.decide_on_probability_threshold(proba_val, n_to_reject)
        return


    def decide_on_probability_threshold(self, prediction_probabilities, n_instances_to_reject):
        ordered_prediction_probs = prediction_probabilities.sort_values(ascending=True)

        if (n_instances_to_reject > len(prediction_probabilities)):
            cut_off_probability = 0.5

        else:
            cut_off_probability = ordered_prediction_probs.iloc[n_instances_to_reject-1]

        return cut_off_probability


    def predict(self, X):
        predictions, probabilities = self.BB.predict_with_proba(X)

        indices_below_uncertainty_threshold = probabilities[probabilities < self.threshold].index

        all_uncertainty_based_rejects_df = pd.DataFrame({
            'instance': [X.descriptive_data.loc[i].to_dict() for i in indices_below_uncertainty_threshold],
            'prediction_without_reject': predictions[indices_below_uncertainty_threshold],
            'prediction probability': probabilities[indices_below_uncertainty_threshold],
        }, index=indices_below_uncertainty_threshold)

        all_uncertainty_based_rejects_series = (pd.Series([]) if all_uncertainty_based_rejects_df.empty
                                                else all_uncertainty_based_rejects_df.apply(
            create_uncertainty_based_reject, axis=1))
        print("UBAC is rejecting: " + str(len(all_uncertainty_based_rejects_series)) + " instances")

        predictions.update(all_uncertainty_based_rejects_series)
        return predictions


class SCross():
    """
    Class for SCrpss
    """

    def __init__(self, base_classifier, coverage, cv=5, seed=42):
        self.cv = cv
        self.seed = seed
        self.cov = coverage
        self.r_ratio = 1 - coverage
        self.theta = 0
        self.base_classifier = base_classifier

    def fit(self, train_data):
        y = train_data.descriptive_data[train_data.decision_attribute]
        X = train_data.one_hot_encoded_data.loc[:,
                  train_data.one_hot_encoded_data.columns != train_data.decision_attribute]

        self.classes_ = unique(y)
        z = []
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
            else:
                X_train = X[train_index]
                X_test = X[test_index]
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_index]
                # y_test = y.iloc[test_index]
            else:
                y_train = y[train_index]
                # y_test = y[test_index]

            # Step 1: Train Black-Box Model
            bb = BlackBoxClassifier(self.base_classifier)
            y_mapping = {train_data.desirable_label: 1, train_data.undesirable_label: 0}
            y_train = y_train.replace(y_mapping)

            trained_bb = bb.fit(X_train, y_train)
            # quantiles
            probas = trained_bb.predict_proba(X_test)
            confs = np.max(probas, axis=1)
            z.append(confs)
        self.z = z
        confs = np.concatenate(z).ravel()
        sub_confs_1, sub_confs_2 = train_test_split(confs, test_size=.5, random_state=42)
        tau = (1 / np.sqrt(2))
        self.theta = (tau * np.quantile(confs, self.r_ratio) + (1 - tau) * (
                    0.5 * np.quantile(sub_confs_1, self.r_ratio) + 0.5 * np.quantile(sub_confs_2, self.r_ratio)))
        print("THETA: " + str(self.theta))
        self.BB = BlackBoxClassifier(self.base_classifier)
        self.BB.fit(X_train, y_train)

    def predict(self, test_data):
        predictions, probabilities = self.BB.predict_with_proba(test_data)

        indices_below_uncertainty_threshold = probabilities[probabilities < self.theta].index

        all_uncertainty_based_rejects_df = pd.DataFrame({
            'instance': [test_data.descriptive_data.loc[i].to_dict() for i in indices_below_uncertainty_threshold],
            'prediction_without_reject': predictions[indices_below_uncertainty_threshold],
            'prediction probability': probabilities[indices_below_uncertainty_threshold],
        }, index=indices_below_uncertainty_threshold)

        all_uncertainty_based_rejects_series = (pd.Series([]) if all_uncertainty_based_rejects_df.empty
                                                else all_uncertainty_based_rejects_df.apply(
                                                create_uncertainty_based_reject, axis=1))

        print("SCross is rejecting: " + str(len(all_uncertainty_based_rejects_series)) + " instances")

        predictions.update(all_uncertainty_based_rejects_series)
        return predictions


class AUCPlugIn():

    def __init__(self, base_classifier, coverage, val_ratio, seed=42):
        self.base_classifier = base_classifier
        self.val_ratio = val_ratio
        self.seed = seed
        self.cov = coverage
        self.r_ratio = 1 - coverage
        self.thetas = tuple()
        self.base_classifier = base_classifier


    def fit(self, train_data):
        y = train_data.descriptive_data[train_data.decision_attribute]
        y_numeric = y.replace({train_data.desirable_label:1, train_data.undesirable_label:0})

        X = train_data.one_hot_encoded_data.loc[:,
            train_data.one_hot_encoded_data.columns != train_data.decision_attribute]

        self.classes_ = unique(y)
        localthetas = []
        X_train, X_hold, y_train, y_hold = train_test_split(X, y_numeric, stratify=y_numeric, random_state=self.seed, test_size=self.val_ratio)
        bb = BlackBoxClassifier(self.base_classifier)
        self.trained_bb = bb.fit(X_train, y_train)

        # quantiles
        y_scores = self.trained_bb.predict_proba(X_hold)[:, 1]
        auc_roc = skm.roc_auc_score(y_hold, y_scores)
        n, npos = len(y_hold), np.sum(y_hold)
        pneg = 1 - np.mean(y_hold)
        u_pos = int(auc_roc * pneg * n)
        pos_sorted = np.argsort(y_scores)
        if isinstance(y_hold, pd.Series):
            tp = np.cumsum(y_hold.iloc[pos_sorted[::-1]])
        else:
            tp = np.cumsum(y_hold[pos_sorted[::-1]])
        l_pos = n - np.searchsorted(tp, auc_roc * npos + 1, side='right')
        # print('Local bounds:', l_pos, '<= rank <=', u_pos, ' pct', (u_pos-l_pos+1)/n)
        # print('Local bounds:', y_scores[pos_sorted[l_pos]], '<= score <=', y_scores[pos_sorted[u_pos]])
        pos = (u_pos + l_pos) / 2
        locallist = []

        delta = int(n * self.r_ratio/2)
        t1 = y_scores[pos_sorted[max(0, round(pos - delta))]]
        t2 = y_scores[pos_sorted[min(round(pos + delta), n - 1)]]
        self.thetas = (t1, t2)


    def predict(self, test_data):
        X = test_data.one_hot_encoded_data.loc[:,
            test_data.one_hot_encoded_data.columns != test_data.decision_attribute]

        predictions_num = pd.Series(self.trained_bb.predict(X))
        predictions = predictions_num.replace({1: test_data.desirable_label, 0: test_data.undesirable_label})

        prediction_probas = pd.Series(self.trained_bb.predict_proba(X)[:, 1])

        t1 = self.thetas[0]
        t2 = self.thetas[1]
        to_reject_indices = prediction_probas[(t1 <= prediction_probas) & (prediction_probas <= t2)].index

        all_uncertainty_based_rejects_df = pd.DataFrame({
            'instance': [test_data.descriptive_data.loc[i].to_dict() for i in to_reject_indices],
            'prediction_without_reject': predictions[to_reject_indices],
            'prediction probability': prediction_probas[to_reject_indices],
        }, index=to_reject_indices)

        all_uncertainty_based_rejects_series = (pd.Series([]) if all_uncertainty_based_rejects_df.empty
                                                else all_uncertainty_based_rejects_df.apply(create_uncertainty_based_reject, axis=1))
        print("AUC is rejecting: " + str(len(all_uncertainty_based_rejects_series)) + " instances")

        predictions.update(all_uncertainty_based_rejects_series)
        return predictions
