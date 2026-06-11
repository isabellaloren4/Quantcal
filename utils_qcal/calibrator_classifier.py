from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from scipy.optimize import minimize_scalar
import numpy as np


class BCTSCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """
    BCTS calibration with an interface identical to scikit-learn's
    CalibratedClassifierCV.

    Parameters
    ----------
    estimator : sklearn classifier
        Base model to be calibrated.
    cv : int or 'prefit', default=5
        - int: number of folds for cross-validation
        - 'prefit': model already trained, uses X for calibration
    reg_lambda : float, default=1e-3
        Regularization of the calibrator.
    cal_size : float, default=0.2
        Size of the calibration set (used only with cv='prefit').
    """

    def __init__(self, estimator, cv=5, reg_lambda=1e-3, cal_size=0.2):
        self.estimator = estimator
        self.cv = cv
        self.reg_lambda = reg_lambda
        self.cal_size = cal_size

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        if self.cv == 'prefit':
            # Model already trained, calibrate with part of the data
            X_cal, _, y_cal, _ = train_test_split(
                X, y,
                test_size=1 - self.cal_size,
                stratify=y,
                random_state=42
            )
            probs_cal = self.estimator.predict_proba(X_cal)
            self.calibrator_ = FullDirichletCalibrator(
                reg_lambda=self.reg_lambda,
                reg_mu=None
            )
            self.calibrator_.fit(probs_cal, y_cal)
            self.estimator_ = self.estimator

        else:
            # Cross-validation: collect out-of-fold probabilities
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            probs_oof = np.zeros((len(y), len(self.classes_)))
            self.estimators_ = []

            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                clf = clone(self.estimator)
                clf.fit(X_tr, y_tr)
                probs_oof[val_idx] = clf.predict_proba(X_val)
                self.estimators_.append(clf)

            # Train the calibrator with all OOF probabilities
            self.calibrator_ = FullDirichletCalibrator(
                reg_lambda=self.reg_lambda,
                reg_mu=None
            )
            self.calibrator_.fit(probs_oof, y)

            # Train the final model on all the data
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y)

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        probs = self.estimator_.predict_proba(X)
        return self.calibrator_.predict_proba(probs)

    def predict(self, X):
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)