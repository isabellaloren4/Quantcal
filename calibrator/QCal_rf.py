from mlquantify.base_aggregative import AggregationMixin
from mlquantify.base import BaseQuantifier
from mlquantify.adjust_counting import CC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils_qcal.app_protocol import *
from utils_qcal.median_estimates import *
from utils_qcal.extract_estimates import *

# # Methods used
# - Regressor: Random Forest
# - Clf: Random Forest Classifier
# - Quantifier: CC
# - Regressor input: Median of CC estimates
# - Variation of splits (3, 5, 10)


class QCal_rf_3(AggregationMixin, BaseQuantifier):
    '''
    Quantifier calibration method.
    - Random Forest regressor for correction
    - Random Forest classifier for the quantifier
    - Median of CC estimates computed for each proportion
    - n_validation = 3 (number of validations for the median computation)
    '''
    def __init__(self, learner, *, n_validation=3, name_data=None):
        self.learner = learner
        self.regressor = RandomForestRegressor()
        self.quantifier = None
        self.n_validation = n_validation
        self.name_data = name_data
        self.subgroups_train = None
        self.subgroups_train_all = None

    def fit(self, X_train, y_train, name_data=None):
        # if nothing is passed here, use the name saved in the constructor
        if name_data is None:
            name_data = self.name_data

        results_class_0 = []
        results_class_1 = []
        true_prevalences_all = []
        for i in range(self.n_validation):
            # Creating validation set
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.3, stratify=y_train
            )
            model_clf = pre_treined_model(X_tr, y_tr)

            # Generating subgroups
            self.subgroups_train = APP_protocol_mlquantify(X_val, y_val)

            # Computing the CC estimates for each subgroup
            true_prevalences, predictions_for_proportion_class_0, predictions_for_proportion_class_1 = \
                extract_cc_estimates_from_train(
                    X_tr, y_tr, self.subgroups_train, model_clf=model_clf)

            # Storing the results of each validation
            results_class_0.append(predictions_for_proportion_class_0)
            results_class_1.append(predictions_for_proportion_class_1)

            # Storing the true proportions of each validation
            true_prevalences_all.extend(true_prevalences)

        # Concatenating the results of all validations for the model prediction
        results_class_0_final = {}
        results_class_1_final = {}
        for key in predictions_for_proportion_class_0.keys():
            results_class_0_final[key] = []
            results_class_1_final[key] = []
            for d0, d1 in zip(results_class_0, results_class_1):
                results_class_0_final[key] += d0[key]
                results_class_1_final[key] += d1[key]

        # Computing the median of the CC estimates for each proportion
        cc_estimates_median = median_cc_estimates_per_proportion(results_class_0_final, results_class_1_final, true_prevalences_all)

        # Fitting the correction regressor on the median CC estimates
        self.regressor.fit(cc_estimates_median, true_prevalences_all)
        self.quantifier = CC(learner=self.learner)
        self.quantifier.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        # Quantifier prediction
        predict_quantifier = extract_cc_estimates_from_test(
            X_test, self.quantifier
        )
        # Correction method prediction
        prevalence = self.regressor.predict(predict_quantifier)

        return prevalence.flatten()


class QCal_rf_5(AggregationMixin, BaseQuantifier):
    '''
    Quantifier calibration method.
    - Random Forest regressor for correction
    - Random Forest classifier for the quantifier
    - Median of CC estimates computed for each proportion
    - n_validation = 5 (number of validations for the median computation)
    '''
    def __init__(self, learner, *, n_validation=5, name_data=None):
        self.learner = learner
        self.regressor = RandomForestRegressor()
        self.quantifier = None
        self.n_validation = n_validation
        self.name_data = name_data
        self.subgroups_train = None
        self.subgroups_train_all = None

    def fit(self, X_train, y_train, name_data=None):
        # if nothing is passed here, use the name saved in the constructor
        if name_data is None:
            name_data = self.name_data

        results_class_0 = []
        results_class_1 = []
        true_prevalences_all = []
        for i in range(self.n_validation):
            # Creating validation set
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.3, stratify=y_train
            )
            model_clf = pre_treined_model(X_tr, y_tr)

            # Generating subgroups
            self.subgroups_train = APP_protocol_mlquantify(X_val, y_val)

            # Computing the CC estimates for each subgroup
            true_prevalences, predictions_for_proportion_class_0, predictions_for_proportion_class_1 = \
                extract_cc_estimates_from_train(
                    X_tr, y_tr, self.subgroups_train, model_clf=model_clf)

            # Storing the results of each validation
            results_class_0.append(predictions_for_proportion_class_0)
            results_class_1.append(predictions_for_proportion_class_1)

            # Storing the true proportions of each validation
            true_prevalences_all.extend(true_prevalences)

        # Concatenating the results of all validations for the model prediction
        results_class_0_final = {}
        results_class_1_final = {}
        for key in predictions_for_proportion_class_0.keys():
            results_class_0_final[key] = []
            results_class_1_final[key] = []
            for d0, d1 in zip(results_class_0, results_class_1):
                results_class_0_final[key] += d0[key]
                results_class_1_final[key] += d1[key]

        # Computing the median of the CC estimates for each proportion
        cc_estimates_median = median_cc_estimates_per_proportion(results_class_0_final, results_class_1_final, true_prevalences_all)

        # Fitting the correction regressor on the median CC estimates
        self.regressor.fit(cc_estimates_median, true_prevalences_all)
        self.quantifier = CC(learner=self.learner)
        self.quantifier.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        # Quantifier prediction
        predict_quantifier = extract_cc_estimates_from_test(
            X_test, self.quantifier
        )
        # Correction method prediction
        prevalence = self.regressor.predict(predict_quantifier)

        return prevalence.flatten()


class QCal_rf_10(AggregationMixin, BaseQuantifier):
    '''
    Quantifier calibration method.
    - Random Forest regressor for correction
    - Random Forest classifier for the quantifier
    - Median of CC estimates computed for each proportion
    - n_validation = 10 (number of validations for the median computation)
    '''
    def __init__(self, learner, *, n_validation=10, name_data=None):
        self.learner = learner
        self.regressor = RandomForestRegressor()
        self.quantifier = None
        self.n_validation = n_validation
        self.name_data = name_data
        self.subgroups_train = None
        self.subgroups_train_all = None

    def fit(self, X_train, y_train, name_data=None):
        # if nothing is passed here, use the name saved in the constructor
        if name_data is None:
            name_data = self.name_data

        results_class_0 = []
        results_class_1 = []
        true_prevalences_all = []
        for i in range(self.n_validation):
            # Creating validation set
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.3, stratify=y_train
            )
            model_clf = pre_treined_model(X_tr, y_tr)

            # Generating subgroups
            self.subgroups_train = APP_protocol_mlquantify(X_val, y_val)

            # Computing the CC estimates for each subgroup
            true_prevalences, predictions_for_proportion_class_0, predictions_for_proportion_class_1 = \
                extract_cc_estimates_from_train(
                    X_tr, y_tr, self.subgroups_train, model_clf=model_clf)

            # Storing the results of each validation
            results_class_0.append(predictions_for_proportion_class_0)
            results_class_1.append(predictions_for_proportion_class_1)

            # Storing the true proportions of each validation
            true_prevalences_all.extend(true_prevalences)

        # Concatenating the results of all validations for the model prediction
        results_class_0_final = {}
        results_class_1_final = {}
        for key in predictions_for_proportion_class_0.keys():
            results_class_0_final[key] = []
            results_class_1_final[key] = []
            for d0, d1 in zip(results_class_0, results_class_1):
                results_class_0_final[key] += d0[key]
                results_class_1_final[key] += d1[key]

        # Computing the median of the CC estimates for each proportion
        cc_estimates_median = median_cc_estimates_per_proportion(results_class_0_final, results_class_1_final, true_prevalences_all)

        # Fitting the correction regressor on the median CC estimates
        self.regressor.fit(cc_estimates_median, true_prevalences_all)
        self.quantifier = CC(learner=self.learner)
        self.quantifier.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        # Quantifier prediction
        predict_quantifier = extract_cc_estimates_from_test(
            X_test, self.quantifier
        )
        # Correction method prediction
        prevalence = self.regressor.predict(predict_quantifier)

        return prevalence.flatten()
