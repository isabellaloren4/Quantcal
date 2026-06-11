from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from mlquantify.adjust_counting import *
from mlquantify.likelihood import *
from mlquantify.mixture import *
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from scipy.stats import entropy
from utils.utils_calibration import *
import pdb

def pre_treined_model(X_train, y_train, clf=None, calibration: bool = False):
    '''
    This function takes a training set and returns a trained classifier.

    ----------
    Parameters:
    ----------
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - clf: Classifier to be trained. Default is *RandomForestClassifier*.
    - *calibration (bool)*: if True, applies probability calibration

    ----------
    Returns:
    ----------
    - If *calibration* is True: calibrated classifier
    - If *calibration* is False: trained classifier
    '''
    if clf is None:
        model_for_scores = RandomForestClassifier(random_state=42, n_jobs=1)
    else:
        model_for_scores = clf

    if calibration:
        calibrated_clf = BCTSCalibratedClassifierCV(estimator=model_for_scores, cv=5)
        calibrated_clf.fit(X_train, y_train)
        return calibrated_clf
    else:
        model_for_scores.fit(X_train, y_train)
        return model_for_scores
    



def extract_cc_estimates_from_train(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Classify and Count (CC) prevalence estimates over the training subgroups.

    For each subgroup produced by the APP protocol, the CC quantifier estimates the class
    prevalences. These estimates are paired with the subgroups' true prevalences and are
    later used to train the QCal correction regressor.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the CC quantifier.

    Returns:
    ----
    - meta_target_train: List of true proportions for each subgroup (regression targets);
    - predictions_for_proportion_class_0: Dictionary with the CC prevalence estimates for class 0;
    - predictions_for_proportion_class_1: Dictionary with the CC prevalence estimates for class 1.
    '''
    models_land = {
        '0': CC
    }

    trained_models_land = {}
    # Training the CC quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    
    proportions_list = []

    predictions_for_proportion_class_0 = {}
    predictions_for_proportion_class_1 = {}

    # Computing the CC estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # CC prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pred_sub = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                pred_sub = [
                    pred_sub.get(0, np.float64(0.0)),
                    pred_sub.get(1, np.float64(0.0))
                ]

        # Grouping the CC estimates by the subgroup's true proportion
        key = tuple(proportions_real)
        if key not in predictions_for_proportion_class_1:
            predictions_for_proportion_class_0[key] = []
            predictions_for_proportion_class_1[key] = []
        predictions_for_proportion_class_0[key].append(pred_sub[0])
        predictions_for_proportion_class_1[key].append(pred_sub[1])
        proportions_list.append(proportions_real)

    true_prevalences = proportions_list

    return true_prevalences, predictions_for_proportion_class_0, predictions_for_proportion_class_1





def extract_cc_estimates_from_test(batch_test, model_trained):
    '''
    Computes the CC prevalence estimate for the test batch using the trained quantifier.

    Parameters:
    ---
    - batch_test: Test data batch;
    - model_trained: Pre-trained quantification method used to produce the estimate.

    Returns:
    ----
    - cc_estimates_test: Array with the CC prevalence estimate for the test batch (shape (1, 2)).
    '''
    sub_x_test = pd.DataFrame(batch_test)
    if 'target' in sub_x_test.columns:
        sub_x_test = sub_x_test.drop(columns=['target'])

    cc_estimates_test = []

    pred = model_trained.predict(sub_x_test)
    # always ensures two classes (0 and 1)
    pred_sub = [
        pred.get(0, np.float64(0.0)),
        pred.get(1, np.float64(0.0))
    ]
    cc_estimates_test.extend(pred_sub)

    cc_estimates_test = np.array(cc_estimates_test).reshape(1, -1)

    return cc_estimates_test