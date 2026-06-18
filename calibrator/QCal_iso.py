from sklearn.isotonic import IsotonicRegression
from mlquantify.base_aggregative import AggregationMixin
from mlquantify.base import BaseQuantifier
from mlquantify.adjust_counting import CC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from utils_qcal.protocol import *
from utils_qcal.median_estimates import *
from utils_qcal.extract_estimates import *

class QCal_isot_3(AggregationMixin, BaseQuantifier):
    '''
    Quantifier calibration method.
    - IsotonicRegression for correction
    - Random Forest classifier for the quantifier
    - Median of CC estimates computed for each proportion
    - n_validation = 3 (number of validations for the median computation)
    - UPP protocol for generating the subgroups
    '''
    def __init__(self, learner, *, n_validation=3, name_data=None):
        self.learner = learner
        self.regressor = IsotonicRegression(
        y_min=0.0,            
        y_max=1.0,           
        increasing=True,      
        out_of_bounds="clip"
        )
        self.quantifier = None
        self.n_validation = n_validation
        self.name_data = name_data
        self.subgroups_train = None
        self.subgroups_train_all = None

    def fit(self, X_train, y_train, name_data=None):
        # if nothing is passed here, use the name saved in the constructor
        if name_data is None:
            name_data = self.name_data

        true_prevalences_all = []
        cc_estimates_all = []
        for i in range(self.n_validation):
            # Creating validation set
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.3, stratify=y_train
            )
            model_clf = pre_treined_model(X_tr, y_tr)

            # Generating subgroups
            self.subgroups_train = UPP_protocol_mlquantify(X_val, y_val)
            # Computing the CC estimates for each subgroup
            cc_estimates, true_prevalences, trained_models_land = \
                extract_cc_estimates_from_train_iso(
                    X_tr, y_tr, self.subgroups_train, model_clf=model_clf)

            # Storing the results of each validation
            cc_estimates_all.extend(cc_estimates)
            
            # Storing the true proportions of each validation
            true_prevalences_all.extend(true_prevalences)
        
        # Fitting the regressor on the CC estimates
        self.quantifier = CC(learner=self.learner)
        self.quantifier.fit(X_train, y_train)
        self.regressor.fit(cc_estimates_all, true_prevalences_all)
        return self

    def predict(self, X_test):
        predict_quantifier = extract_cc_estimates_from_test(X_test, self.quantifier)

        cc_class1 = predict_quantifier[:, 1]
        prev_class1 = self.regressor.predict(cc_class1)[0]
        prev_class0 = 1.0 - prev_class1

        
        return np.array([prev_class0, prev_class1])