from sklearn.ensemble import RandomForestClassifier
from quapy.method.aggregative import KDEyML
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from mlquantify.adjust_counting import *
from mlquantify.likelihood import *
from mlquantify.mixture import *
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from utils_qcal.calibrator_classifier import *
import pdb

def pre_treined_model(X_train, y_train, clf=None, method=None, calibration: bool = False):
    '''
    This function takes a training set and returns a trained classifier.

    ----------
    Parameters:
    ----------
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - clf: Classifier to be trained. Default is *RandomForestClassifier*.
    - *calibration (bool)*: if True, applies probability calibration.
    - *method (str)*: the calibration method to use. One of 'BCTS',
      'platt_scaling' or 'isotonic'. Default is 'BCTS'.

    ----------
    Returns:
    ----------
    - If *calibration* is True: calibrated classifier.
    - If *calibration* is False: trained classifier.

    ----------
    Raises:
    ----------
    - ValueError: if calibration is requested and *method* is not one of
      'BCTS', 'platt_scaling' or 'isotonic'.
    '''
    if clf is None:
        model_for_scores = RandomForestClassifier(random_state=42, n_jobs=1)
    else:
        model_for_scores = clf

    if calibration or method is not None:
        # default to 'BCTS' when calibration is requested without a method
        if method is None:
            method = 'BCTS'

        if method == 'BCTS':
            calibrated_clf = BCTSCalibratedClassifierCV(estimator=model_for_scores, cv=5)
        elif method == 'platt_scaling':
            calibrated_clf = CalibratedClassifierCV(estimator=model_for_scores, cv=5, method='sigmoid')
        elif method == 'isotonic':
            calibrated_clf = CalibratedClassifierCV(estimator=model_for_scores, cv=5, method='isotonic')
        else:
            raise ValueError(f"Unknown calibration method: {method!r}")

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





# def extract_estimates_from_test(batch_test, model_trained):
#     '''
#     Computes the prevalence estimate for the test batch using the trained quantifier.

#     Parameters:
#     ---
#     - batch_test: Test data batch;
#     - model_trained: Pre-trained quantification method used to produce the estimate.

#     Returns:
#     ----
#     - q_estimates_test: Array with the prevalence estimate for the test batch (shape (1, 2)).
#     '''
#     sub_x_test = pd.DataFrame(batch_test)
#     if 'target' in sub_x_test.columns:
#         sub_x_test = sub_x_test.drop(columns=['target'])

#     q_estimates_test = []

#     pred = model_trained.predict(sub_x_test)
#     # always ensures two classes (0 and 1)
#     pred_sub = [
#         pred.get(0, np.float64(0.0)),
#         pred.get(1, np.float64(0.0))
#     ]
#     q_estimates_test.extend(pred_sub)

#     q_estimates_test = np.array(q_estimates_test).reshape(1, -1)

#     return q_estimates_test



def extract_estimates_from_test(batch_test, model_trained):
    '''
    Computes the prevalence estimate for the test batch using the trained quantifier.
    Handles both mlquantify (dict {0: p0, 1: p1}) and QuaPy (np.ndarray [p0, p1]) outputs.

    Parameters:
    ---
    - batch_test: Test data batch;
    - model_trained: Pre-trained quantification method used to produce the estimate.

    Returns:
    ----
    - q_estimates_test: Array with the prevalence estimate for the test batch (shape (1, 2)).
    '''
    sub_x_test = pd.DataFrame(batch_test)
    if 'target' in sub_x_test.columns:
        sub_x_test = sub_x_test.drop(columns=['target'])

    pred = model_trained.predict(sub_x_test)

    if isinstance(pred, dict):           # mlquantify
        pred_sub = [pred.get(0, np.float64(0.0)), pred.get(1, np.float64(0.0))]
    else:                                # QuaPy (np.ndarray [prev_0, prev_1])
        pred_sub = np.asarray(pred).ravel()

    q_estimates_test = np.array(pred_sub, dtype=float).reshape(1, -1)

    return q_estimates_test



def extract_cc_estimates_from_train_mlq(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Classify and Count (CC) prevalence estimates over the training subgroups
    (flat version used by MLQ; also returns the trained CC quantifier).

    For each subgroup produced by the APP protocol, the CC quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_0, class_1] pair per
    subgroup), together with the subgroups' true prevalences and the trained CC quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the CC quantifier.

    Returns:
    ----
    - cc_estimates: List of CC prevalence estimates, one [class_0, class_1] pair per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained CC quantifier (key '0').
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

    true_prevalences = []
    cc_estimates = []

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
            cc_estimates.append(pred_sub)

        true_prevalences.append(proportions_real)

    return cc_estimates, true_prevalences, trained_models_land




def extract_emq_estimates_from_train(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Expectation-Maximization Quantifier (EMQ) prevalence estimates over the training subgroups
    (flat version used by MLQ; also returns the trained CC quantifier).

    For each subgroup produced by the UPP protocol, the EMQ quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_0, class_1] pair per
    subgroup), together with the subgroups' true prevalences and the trained EMQ quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the EMQ quantifier.

    Returns:
    ----
    - emq_estimates: List of EMQ prevalence estimates, one [class_0, class_1] pair per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained EMQ quantifier (key '0').
    '''
    models_land = {
        '0': EMQ
    }

    trained_models_land = {}
    # Training the EMQ quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    emq_estimates = []

    # Computing the EMQ estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # EMQ prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                emq_pred = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                emq_pred = [
                    emq_pred.get(0, np.float64(0.0)),
                    emq_pred.get(1, np.float64(0.0))
                ]
            emq_estimates.append(emq_pred)

        true_prevalences.append(proportions_real)

    return emq_estimates, true_prevalences


def extract_ac_estimates_from_train(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Adjusted Count (AC) prevalence estimates over the training subgroups
    (flat version used by MLQ; also returns the trained AC quantifier).

    For each subgroup produced by the UPP protocol, the AC quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_0, class_1] pair per
    subgroup), together with the subgroups' true prevalences and the trained AC quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the AC quantifier.

    Returns:
    ----
    - ac_estimates: List of AC prevalence estimates, one [class_0, class_1] pair per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained AC quantifier (key '0').
    '''
    models_land = {
        '0': AC
    }

    trained_models_land = {}
    # Training the AC quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    ac_estimates = []

    # Computing the AC estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # AC prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                ac_pred = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                ac_pred = [
                    ac_pred.get(0, np.float64(0.0)),
                    ac_pred.get(1, np.float64(0.0))
                ]
            ac_estimates.append(ac_pred)

        true_prevalences.append(proportions_real)

    return ac_estimates, true_prevalences, trained_models_land


def extract_pac_estimates_from_train(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Probabilistic Adjusted Count (PAC) prevalence estimates over the training subgroups
    (flat version used by MLQ; also returns the trained PAC quantifier).

    For each subgroup produced by the UPP protocol, the PAC quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_0, class_1] pair per
    subgroup), together with the subgroups' true prevalences and the trained PAC quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the PAC quantifier.

    Returns:
    ----
    - pac_estimates: List of PAC prevalence estimates, one [class_0, class_1] pair per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained PAC quantifier (key '0').
    '''
    models_land = {
        '0': PAC
    }

    trained_models_land = {}
    # Training the PAC quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    pac_estimates = []

    # Computing the PAC estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # PAC prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pac_pred = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                pac_pred = [
                    pac_pred.get(0, np.float64(0.0)),
                    pac_pred.get(1, np.float64(0.0))
                ]
            pac_estimates.append(pac_pred)

        true_prevalences.append(proportions_real)

    return pac_estimates, true_prevalences, trained_models_land


def extract_kdeyml_estimates_from_train(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Kernel Density Estimation (KDEyML) prevalence estimates over the training subgroups
    (flat version used by MLQ; also returns the trained KDEyML quantifier).

    For each subgroup produced by the UPP protocol, the KDEyML quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_0, class_1] pair per
    subgroup), together with the subgroups' true prevalences and the trained KDEyML quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base (already fitted) classifier used by the KDEyML quantifier.

    Returns:
    ----
    - kdeyml_estimates: List of KDEyML prevalence estimates, one [class_0, class_1] pair per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained KDEyML quantifier (key '0').
    '''
    models_land = {
        '0': KDEyML
    }

    trained_models_land = {}
    # Training the KDEyML quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(classifier=model_clf, fit_classifier=False)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    kdeyml_estimates = []

    # Computing the KDEyML estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # KDEyML prevalence estimate for the subgroup (QuaPy returns np.ndarray [prev_0, prev_1])
        for name, model in trained_models_land.items():
            if name == '0':
                kdeyml_pred = model.predict(sub_x)
                kdeyml_pred = [np.float64(kdeyml_pred[0]), np.float64(kdeyml_pred[1])]
            kdeyml_estimates.append(kdeyml_pred)

        true_prevalences.append(proportions_real)

    return kdeyml_estimates, true_prevalences, trained_models_land


def extract_fm_estimates_from_train(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Friedman (FM) prevalence estimates over the training subgroups
    (flat version used by MLQ; also returns the trained FM quantifier).

    For each subgroup produced by the UPP protocol, the FM quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_0, class_1] pair per
    subgroup), together with the subgroups' true prevalences and the trained FM quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the FM quantifier.

    Returns:
    ----
    - fm_estimates: List of FM prevalence estimates, one [class_0, class_1] pair per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained FM quantifier (key '0').
    '''
    models_land = {
        '0': FM
    }

    trained_models_land = {}
    # Training the FM quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    fm_estimates = []

    # Computing the FM estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # FM prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                fm_pred = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                fm_pred = [
                    fm_pred.get(0, np.float64(0.0)),
                    fm_pred.get(1, np.float64(0.0))
                ]
            fm_estimates.append(fm_pred)

        true_prevalences.append(proportions_real)

    return fm_estimates, true_prevalences, trained_models_land


def extract_dys_estimates_from_train(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Distribution y-Similarity (DyS) prevalence estimates over the training subgroups
    (flat version used by MLQ; also returns the trained DyS quantifier).

    For each subgroup produced by the UPP protocol, the DyS quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_0, class_1] pair per
    subgroup), together with the subgroups' true prevalences and the trained DyS quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the DyS quantifier.

    Returns:
    ----
    - dys_estimates: List of DyS prevalence estimates, one [class_0, class_1] pair per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained DyS quantifier (key '0').
    '''
    models_land = {
        '0': DyS
    }

    trained_models_land = {}
    # Training the DyS quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    dys_estimates = []

    # Computing the DyS estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # DyS prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                dys_pred = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                dys_pred = [
                    dys_pred.get(0, np.float64(0.0)),
                    dys_pred.get(1, np.float64(0.0))
                ]
            dys_estimates.append(dys_pred)

        true_prevalences.append(proportions_real)

    return dys_estimates, true_prevalences, trained_models_land


def extract_ms_estimates_from_train(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Median Sweep (MS) prevalence estimates over the training subgroups
    (flat version used by MLQ; also returns the trained MS quantifier).

    For each subgroup produced by the UPP protocol, the MS quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_0, class_1] pair per
    subgroup), together with the subgroups' true prevalences and the trained MS quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the MS quantifier.

    Returns:
    ----
    - ms_estimates: List of MS prevalence estimates, one [class_0, class_1] pair per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained MS quantifier (key '0').
    '''
    models_land = {
        '0': MS
    }

    trained_models_land = {}
    # Training the MS quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    ms_estimates = []

    # Computing the MS estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # MS prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                ms_pred = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                ms_pred = [
                    ms_pred.get(0, np.float64(0.0)),
                    ms_pred.get(1, np.float64(0.0))
                ]
            ms_estimates.append(ms_pred)

        true_prevalences.append(proportions_real)

    return ms_estimates, true_prevalences, trained_models_land


def extract_cc_estimates_from_train_iso(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Classify and Count (CC) prevalence estimates over the training subgroups
    (flat version used by ISO; also returns the trained CC quantifier).

    For each subgroup produced by the protocol, the CC quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_1] per
    subgroup), together with the subgroups' true prevalences and the trained CC quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the CC quantifier.

    Returns:
    ----
    - cc_estimates: List of CC prevalence estimates, one [class_1] per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained CC quantifier (key '0').
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

    true_prevalences = []
    cc_estimates = []

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
            cc_estimates.append(pred_sub[1]) # only the prevalence of class 1 is needed for the regression

        true_prevalences.append(proportions_real[1]) # only the prevalence of class 1 is needed for the regression

    return cc_estimates, true_prevalences, trained_models_land





def extract_dys_estimates_from_train_iso(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Distribution y-Similarity (DyS) prevalence estimates over the training subgroups
    (flat version used by ISO; also returns the trained DyS quantifier).

    For each subgroup produced by the protocol, the DyS quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_1] per
    subgroup), together with the subgroups' true prevalences and the trained DyS quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the DyS quantifier.

    Returns:
    ----
    - dyS_estimates: List of DyS prevalence estimates, one [class_1] per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained DyS quantifier (key '0').
    '''
    models_land = {
        '0': DyS
    }

    trained_models_land = {}
    # Training the DyS quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    dyS_estimates = []

    # Computing the DyS estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # DyS prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pred_sub = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                pred_sub = [
                    pred_sub.get(0, np.float64(0.0)),
                    pred_sub.get(1, np.float64(0.0))
                ]
            dyS_estimates.append(pred_sub[1]) # only the prevalence of class 1 is needed for the regression

        true_prevalences.append(proportions_real[1]) # only the prevalence of class 1 is needed for the regression

    return dyS_estimates, true_prevalences



def extract_emq_estimates_from_train_iso(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Expectation-Maximization Quantifier (EMQ) prevalence estimates over the training subgroups
    (flat version used by ISO; also returns the trained EMQ quantifier).

    For each subgroup produced by the protocol, the EMQ quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_1] per
    subgroup), together with the subgroups' true prevalences and the trained EMQ quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the EMQ quantifier.

    Returns:
    ----
    - emq_estimates: List of EMQ prevalence estimates, one [class_1] per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained EMQ quantifier (key '0').
    '''
    models_land = {
        '0': EMQ
    }

    trained_models_land = {}
    # Training the EMQ quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    emq_estimates = []

    # Computing the EMQ estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # EMQ prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pred_sub = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                pred_sub = [
                    pred_sub.get(0, np.float64(0.0)),
                    pred_sub.get(1, np.float64(0.0))
                ]
            emq_estimates.append(pred_sub[1]) # only the prevalence of class 1 is needed for the regression

        true_prevalences.append(proportions_real[1]) # only the prevalence of class 1 is needed for the regression

    return emq_estimates, true_prevalences



def extract_ms_estimates_from_train_iso(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Median Sweep (MS) prevalence estimates over the training subgroups
    (flat version used by ISO; also returns the trained MS quantifier).

    For each subgroup produced by the protocol, the MS quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_1] per
    subgroup), together with the subgroups' true prevalences and the trained MS quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the MS quantifier.

    Returns:
    ----
    - ms_estimates: List of MS prevalence estimates, one [class_1] per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained MS quantifier (key '0').
    '''
    models_land = {
        '0': MS
    }

    trained_models_land = {}
    # Training the MS quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    ms_estimates = []

    # Computing the MS estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # MS prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pred_sub = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                pred_sub = [
                    pred_sub.get(0, np.float64(0.0)),
                    pred_sub.get(1, np.float64(0.0))
                ]
            ms_estimates.append(pred_sub[1]) # only the prevalence of class 1 is needed for the regression

        true_prevalences.append(proportions_real[1]) # only the prevalence of class 1 is needed for the regression

    return ms_estimates, true_prevalences



def extract_fm_estimates_from_train_iso(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Friedman (2015) prevalence estimates over the training subgroups
    (flat version used by ISO; also returns the trained FM quantifier).

    For each subgroup produced by the protocol, the FM quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_1] per
    subgroup), together with the subgroups' true prevalences and the trained FM quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the FM quantifier.

    Returns:
    ----
    - fm_estimates: List of FM prevalence estimates, one [class_1] per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained FM quantifier (key '0').
    '''
    models_land = {
        '0': FM
    }

    trained_models_land = {}
    # Training the FM quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    fm_estimates = []

    # Computing the FM estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # FM prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pred_sub = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                pred_sub = [
                    pred_sub.get(0, np.float64(0.0)),
                    pred_sub.get(1, np.float64(0.0))
                ]
            fm_estimates.append(pred_sub[1]) # only the prevalence of class 1 is needed for the regression

        true_prevalences.append(proportions_real[1]) # only the prevalence of class 1 is needed for the regression

    return fm_estimates, true_prevalences




def extract_ac_estimates_from_train_iso(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Adjusted Count (AC) prevalence estimates over the training subgroups
    (flat version used by ISO; also returns the trained AC quantifier).

    For each subgroup produced by the protocol, the AC quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_1] per
    subgroup), together with the subgroups' true prevalences and the trained AC quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the AC quantifier.

    Returns:
    ----
    - ac_estimates: List of AC prevalence estimates, one [class_1] per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained AC quantifier (key '0').
    '''
    models_land = {
        '0': AC
    }

    trained_models_land = {}
    # Training the AC quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    ac_estimates = []

    # Computing the AC estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # AC prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pred_sub = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                pred_sub = [
                    pred_sub.get(0, np.float64(0.0)),
                    pred_sub.get(1, np.float64(0.0))
                ]
            ac_estimates.append(pred_sub[1]) # only the prevalence of class 1 is needed for the regression

        true_prevalences.append(proportions_real[1]) # only the prevalence of class 1 is needed for the regression

    return ac_estimates, true_prevalences





def extract_pac_estimates_from_train_iso(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Probabilistic Adjusted Count (PAC) prevalence estimates over the training subgroups
    (flat version used by ISO; also returns the trained PAC quantifier).

    For each subgroup produced by the protocol, the PAC quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_1] per
    subgroup), together with the subgroups' true prevalences and the trained PAC quantifier.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base classifier used by the PAC quantifier.

    Returns:
    ----
    - pac_estimates: List of PAC prevalence estimates, one [class_1] per subgroup;
    - true_prevalences: List of true proportions for each subgroup;
    - trained_models_land: Dictionary with the trained PAC quantifier (key '0').
    '''
    models_land = {
        '0': PAC
    }

    trained_models_land = {}
    # Training the PAC quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(learner=model_clf)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    pac_estimates = []

    # Computing the PAC estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # PAC prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pred_sub = model.predict(sub_x)
                # always ensures two classes (0 and 1)
                pred_sub = [
                    pred_sub.get(0, np.float64(0.0)),
                    pred_sub.get(1, np.float64(0.0))
                ]
            pac_estimates.append(pred_sub[1]) # only the prevalence of class 1 is needed for the regression

        true_prevalences.append(proportions_real[1]) # only the prevalence of class 1 is needed for the regression

    return pac_estimates, true_prevalences






def extract_hdx_estimates_from_train_iso(X_train, y_train, subgroups):
    '''
    Computes the HDx prevalence estimates over the training subgroups
    (flat version used by ISO).

    HDx is non-aggregative: it uses feature-space histograms and does not
    require a classifier. The model is fitted once on X_train/y_train and
    then applied to each subgroup.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples (batch, true_proportions) from UPP protocol.

    Returns:
    ----
    - hdx_estimates: List of HDx prevalence estimates, one scalar (class_1) per subgroup;
    - true_prevalences: List of true proportions (class_1) for each subgroup.
    '''
    model = HDx()
    model.fit(X_train, y_train)

    true_prevalences = []
    hdx_estimates = []

    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        pred_sub = model.predict(sub_x)
        pred_sub = [
            pred_sub.get(0, np.float64(0.0)),
            pred_sub.get(1, np.float64(0.0))
        ]
        hdx_estimates.append(pred_sub[1])
        true_prevalences.append(proportions_real[1])

    return hdx_estimates, true_prevalences


def _build_and_fit_quantifier(quantifier_cls, X, y, model_clf=None, fit_classifier=True):
    '''
    Instantiates and fits `quantifier_cls`, adapting the constructor call to
    the specific class in use:
    - KDEyML expects `classifier=` (+ `fit_classifier`, see below);
    - HDx is non-aggregative and takes no classifier at all;
    - every other quantifier (CC, AC, PAC, DyS, MS, EMQ, FM, ...) expects
      `learner=`.

    `fit_classifier` only matters for KDEyML:
    - Pass False when `model_clf` is already a trained classifier (e.g. the
      one built by `pre_treined_model` during QCal's validation loop) — the
      classifier should not be retrained.
    - Pass True (the default) when `model_clf` is a fresh, untrained learner
      (e.g. `self.learner` when building QCal's final quantifier) — KDEyML
      needs to fit it itself.

    This is what lets extract_estimates_from_train_iso work for every
    method with a single implementation instead of one copy-pasted
    function per quantifier.
    '''
    if quantifier_cls is KDEyML:
        model = quantifier_cls(classifier=model_clf, fit_classifier=fit_classifier)
    elif quantifier_cls is HDx:
        model = quantifier_cls()
    else:
        model = quantifier_cls(learner=model_clf)

    model.fit(X, y)
    return model


def _predict_class1(model, sub_x):
    '''
    Returns the predicted prevalence of class 1 for `sub_x`, handling both
    mlquantify's dict output ({0: p0, 1: p1}) and QuaPy's ndarray output
    ([p0, p1], used by KDEyML).
    '''
    pred = model.predict(sub_x)
    if isinstance(pred, dict):
        return pred.get(1, np.float64(0.0))
    return np.asarray(pred).ravel()[1]


def extract_estimates_from_train_iso(X_train, y_train, subgroups, quantifier_cls, model_clf=None):
    '''
    Generic (ISO) prevalence-estimate extraction, parameterized by the
    quantifier class instead of hard-coded per method.

    This single function replaces extract_cc_estimates_from_train_iso,
    extract_dys_estimates_from_train_iso, extract_emq_estimates_from_train_iso,
    extract_ms_estimates_from_train_iso, extract_fm_estimates_from_train_iso,
    extract_ac_estimates_from_train_iso, extract_pac_estimates_from_train_iso,
    extract_kde_estimates_from_train_iso and extract_hdx_estimates_from_train_iso
    (which only differed in which quantifier class they trained). Those
    functions are kept below, untouched, for backward compatibility with any
    other code that still calls them directly.

    Trains `quantifier_cls` once on (X_train, y_train) and computes its
    class-1 prevalence estimate for every subgroup, pairing it with the
    subgroup's true class-1 prevalence.

    Parameters
    ----------
    - X_train, y_train: training data used to fit the quantifier;
    - subgroups: list of (batch, true_proportions) tuples, as produced by the
      UPP/APP protocol;
    - quantifier_cls: the quantifier class to use
      (CC, AC, PAC, DyS, MS, EMQ, FM, KDEyML or HDx);
    - model_clf: pre-trained base classifier forwarded to the quantifier
      (ignored for HDx, which is non-aggregative). Assumed already fitted —
      this function always calls _build_and_fit_quantifier with
      fit_classifier=False.

    Returns
    ----
    - estimates: list of class-1 prevalence estimates, one per subgroup;
    - true_prevalences: list of true class-1 prevalences, one per subgroup.
    '''
    model = _build_and_fit_quantifier(quantifier_cls, X_train, y_train, model_clf=model_clf, fit_classifier=False)

    estimates = []
    true_prevalences = []
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        estimates.append(_predict_class1(model, sub_x))
        true_prevalences.append(proportions_real[1])

    return estimates, true_prevalences


def extract_estimates_from_test_subgroups(subgroups, model):
    '''
    Generic prevalence-estimate extraction over *test* subgroups, using a
    quantifier that is ALREADY fitted.

    This is the held-out counterpart of `extract_estimates_from_train_iso`.
    The crucial difference is that this function does NOT fit anything: it
    receives the quantifier already trained on the full training set (e.g.
    `QCal.quantifier` after `QCal.fit()`) and only reads off its class-1
    estimate for each test subgroup. That is what makes the resulting pairs
    genuinely out-of-sample: neither the quantifier nor the isotonic
    regressor has seen these subgroups.

    Parameters
    ----------
    - subgroups: list of (batch, true_proportions) tuples, as produced by the
      UPP/APP protocol run on the TEST set;
    - model: an already-fitted quantifier (CC, AC, PAC, DyS, MS, EMQ, FM,
      KDEyML or HDx instance).

    Returns
    ----
    - estimates: list of class-1 prevalence estimates, one per test subgroup;
    - true_prevalences: list of true class-1 prevalences, one per test subgroup.
    '''
    estimates = []
    true_prevalences = []
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x)
        if 'target' in sub_x.columns:
            sub_x = sub_x.drop(columns=['target'])
        estimates.append(_predict_class1(model, sub_x))
        true_prevalences.append(proportions_real[1])

    return estimates, true_prevalences


def extract_kde_estimates_from_train_iso(X_train, y_train, subgroups, model_clf):
    '''
    Computes the Kernel Density Estimation model for quantification (KDEy) prevalence estimates over the training subgroups
    (flat version used by ISO).

    For each subgroup produced by the protocol, the KDEy quantifier estimates the class
    prevalences. The estimates are returned as a flat list (one [class_1] per
    subgroup), together with the subgroups' true prevalences.

    Parameters:
    ---
    - X_train: Training dataset;
    - y_train: Labels of the training dataset;
    - subgroups: List of tuples, where each tuple contains a batch and the true class proportions in that batch;
    - model_clf: Base (already fitted) classifier used by the KDEy quantifier.

    Returns:
    ----
    - kde_estimates: List of KDEy prevalence estimates, one [class_1] per subgroup;
    - true_prevalences: List of true proportions for each subgroup.
    '''
    models_land = {
        '0': KDEyML
    }

    trained_models_land = {}
    # Training the KDEy quantifier used to produce the estimates
    for name, model_class in models_land.items():
        model = model_class(classifier=model_clf, fit_classifier=False)
        model.fit(X_train, y_train)
        trained_models_land[name] = model

    true_prevalences = []
    kde_estimates = []

    # Computing the KDEy estimates for each training subgroup
    for sub_x, proportions_real in subgroups:
        sub_x = pd.DataFrame(sub_x).drop(columns=['target'])
        # KDEy prevalence estimate for the subgroup
        for name, model in trained_models_land.items():
            if name == '0':
                pred_sub = model.predict(sub_x)  # QuaPy returns np.ndarray [prev_0, prev_1]
                kde_estimates.append(pred_sub[1])  # only the prevalence of class 1 is needed for the regression

        true_prevalences.append(proportions_real[1])  # only the prevalence of class 1 is needed for the regression

    return kde_estimates, true_prevalences