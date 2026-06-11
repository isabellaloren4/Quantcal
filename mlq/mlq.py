from mlquantify.base_aggregative import AggregationMixin
from mlquantify.base import BaseQuantifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from utils_qcal.app_protocol import *
from utils_qcal.calibrator_classifier import *
from utils_qcal.extract_estimates import *


class MLQ_rf(AggregationMixin, BaseQuantifier):
    def __init__(self, learner, *, n_validation=1, name_data=None):
        self.learner = learner
        self.regressor = MultiOutputRegressor(LinearSVR())
        self.n_validation = n_validation
        self.name_data = name_data
        self.train = None
        self.subgroups_train = None
        self.subgroups_train_all = None
        self.results_class_1_final = None

    def fit(self, X_train, y_train, name_data=None):
        # if nothing is passed here, use the name saved in the constructor
        if name_data is None:
            name_data = self.name_data

        true_prevalences_all = []
        cc_estimates_all = []
        for i in range(self.n_validation):
            # Creating validation set
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.4, stratify=y_train
            )
            model_clf = pre_treined_model(X_tr, y_tr)

            # Generating subgroups
            self.subgroups_train = APP_protocol_mlquantify(X_val, y_val)

            # Computing the CC estimates for each subgroup
            cc_estimates, true_prevalences, trained_models_land = \
                extract_cc_estimates_from_train_mlq(
                    X_tr, y_tr, self.subgroups_train, model_clf=model_clf)

            # Storing the results of each validation
            cc_estimates_all.extend(cc_estimates)

            # Storing the true proportions of each validation
            true_prevalences_all.extend(true_prevalences)

        # Fitting the regressor on the CC estimates
        self.quantifier = trained_models_land['0']
        self.regressor.fit(cc_estimates_all, true_prevalences_all)
        return self

    def predict(self, X_test):
        # Quantifier prediction
        predict_quantifier = extract_cc_estimates_from_test(
            X_test, self.quantifier
        )
        # Correction method prediction
        prevalence = self.regressor.predict(predict_quantifier)

        prevalence = np.clip(prevalence, 0, 1)  # ensure predictions are between 0 and 1

        return prevalence.flatten()


class MLQ_rf_calib(AggregationMixin, BaseQuantifier):
    def __init__(self, learner, *, n_validation=1, name_data=None):
        self.learner = learner
        self.regressor = MultiOutputRegressor(LinearSVR())
        self.n_validation = n_validation
        self.name_data = name_data
        self.train = None
        self.subgroups_train = None
        self.subgroups_train_all = None
        self.results_class_1_final = None

    def fit(self, X_train, y_train, name_data=None):
        # if nothing is passed here, use the name saved in the constructor
        if name_data is None:
            name_data = self.name_data

        true_prevalences_all = []
        cc_estimates_all = []
        for i in range(self.n_validation):
            # Creating validation set
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.4, stratify=y_train
            )
            model_clf = pre_treined_model(X_tr, y_tr, calibration=True)

            # Generating subgroups
            self.subgroups_train = APP_protocol_mlquantify(X_val, y_val)

            # Computing the CC estimates for each subgroup
            cc_estimates, true_prevalences, trained_models_land = \
                extract_cc_estimates_from_train_mlq(
                    X_tr, y_tr, self.subgroups_train, model_clf=model_clf)

            # Storing the results of each validation
            cc_estimates_all.extend(cc_estimates)

            # Storing the true proportions of each validation
            true_prevalences_all.extend(true_prevalences)

        # Fitting the regressor on the CC estimates
        self.quantifier = trained_models_land['0']
        self.regressor.fit(cc_estimates_all, true_prevalences_all)
        return self

    def predict(self, X_test):
        # Quantifier prediction
        predict_quantifier = extract_cc_estimates_from_test(
            X_test, self.quantifier
        )
        # Correction method prediction
        prevalence = self.regressor.predict(predict_quantifier)

        prevalence = np.clip(prevalence, 0, 1)  # ensure predictions are between 0 and 1

        return prevalence.flatten()
