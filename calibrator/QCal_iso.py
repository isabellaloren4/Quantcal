from sklearn.isotonic import IsotonicRegression
import os
from mlquantify.base_aggregative import AggregationMixin
from mlquantify.base import BaseQuantifier
from mlquantify.adjust_counting import CC
from mlquantify.likelihood import *
from mlquantify.mixture import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from utils_qcal.protocol import *
from utils_qcal.median_estimates import *
from utils_qcal.extract_estimates import *
from utils_qcal.extract_estimates import _build_and_fit_quantifier  # not exported by `import *` (leading underscore)
from utils_qcal.qcal_plots import (
    plot_estimates_vs_real,
    plot_isotonic_calibration,
    plot_test_calibration,
    plot_test_isotonic_curve,
)



from sklearn.isotonic import IsotonicRegression
import os
from mlquantify.base_aggregative import AggregationMixin
from mlquantify.base import BaseQuantifier
from mlquantify.adjust_counting import CC
from mlquantify.likelihood import *
from mlquantify.mixture import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from utils_qcal.protocol import *
from utils_qcal.median_estimates import *
from utils_qcal.extract_estimates import *
from utils_qcal.extract_estimates import _build_and_fit_quantifier  # not exported by `import *` (leading underscore)
from utils_qcal.qcal_plots import (
    plot_estimates_vs_real,
    plot_isotonic_calibration,
    plot_test_calibration,
    plot_test_isotonic_curve,
)


class QCal(AggregationMixin, BaseQuantifier):
    '''
    Unified Quantifier Calibration (QCal) method.

    Instead of one hand-written class per base quantifier
    (QCal_isot_3, QCal_isot_3_dys, QCal_isot_3_kde, ...), this single
    class covers all of them: pick the base quantifier via
    `method_name` and everything else (extraction routine, quantifier
    instantiation) is resolved automatically from `_REGISTRY`.

    - IsotonicRegression for the post-hoc calibration
    - `n_validation` controls the number of validation splits used to
      build the (estimate, true_prevalence) pairs fed to the regressor
      (this replaces the old "_3" / "_10" suffixes: just pass
      n_validation=3 or n_validation=10, or any other value).
    - UPP protocol for generating the subgroups

    Parameters
    ----------
    learner : classifier
        Base learner passed to the underlying quantifier (ignored for
        method_name='hdx', which does not use a classifier).
    method_name : str
        Which base quantifier/estimator to calibrate. One of:
        'cc', 'acc', 'pac', 'dys', 'ms', 'emq', 'fm', 'kde', 'hdx'.
    clf : classifier
        Classifier used by `pre_treined_model` to build the model
        used during the estimate-extraction step (not used for 'hdx').
    n_validation : int, default=3
        Number of validation splits (previously hard-coded as 3 or 10
        in separate classes).
    n_prevalences : int or None, default=None
        Number of prevalence points used by UPP_protocol_mlquantify to build
        each validation split's subgroups. None keeps UPP's own default
        (1100); set e.g. n_prevalences=100 to use fewer, coarser subgroups.
    method : str or None, default=None
        Calibration method forwarded to `pre_treined_model`'s `method`
        argument — one of 'BCTS', 'platt_scaling' or 'isotonic'. This is
        the *classifier* calibration method (Platt/BCTS/isotonic scaling
        of `clf`'s scores), not to be confused with `method_name`, which
        selects the *quantifier* (cc, acc, kde, ...) that QCal itself
        calibrates.
    calibrator : bool, default=False
        Forwarded to `pre_treined_model`'s `calibration` argument. Despite
        the name, this is a flag, not a calibrator object/instance.
        - If True: `clf` is wrapped in a calibrated classifier (`method`,
          defaulting to 'BCTS' if not given).
        - If False but `method` is set: `pre_treined_model` calibrates
          anyway — its internal check is `calibration or method is not
          None`, so setting `method` alone is enough to trigger
          calibration even with `calibrator=False`.
        - Only when both `calibrator=False` and `method=None` is `clf`
          trained plain, uncalibrated.
        Leave both at their defaults (`calibrator=False`, `method=None`)
        for a plain classifier; set `method` (with or without
        `calibrator=True` — the effect is the same) to calibrate it.
    plot : bool, default=False
        If True, fit() also builds the two diagnostic plots
        (`plot_estimates()` and `plot_calibration()`) and stores them in
        `self.fig_estimates_` / `self.fig_calibration_`. Kept off by
        default so it doesn't interfere with large parallel experiment
        runs (e.g. ProcessPoolExecutor across many datasets/splits) —
        call `.plot_estimates()` / `.plot_calibration()` manually when you
        want a figure for a specific run.
    plot_dir : str or Path or None, default=None
        Only used when plot=True. Directory where fit() automatically
        saves the two diagnostic figures. Filenames are built as
        '{name_data}_{method_name}_estimates.png' and
        '{name_data}_{method_name}_calibration.png' (falling back to
        'data' in place of name_data if none was given). The directory is
        created automatically if it doesn't exist. If plot_dir is None
        (default), figures are still built when plot=True but nothing is
        saved to disk unless you pass estimates_plot_path /
        calibration_plot_path explicitly to fit() (those take precedence
        over plot_dir if both are given).

    Examples
    --------
    >>> QCal(learner=RandomForestClassifier(), method_name='cc', clf=clf, n_validation=3)
    >>> QCal(learner=RandomForestClassifier(), method_name='kde', clf=clf, n_validation=10)
    >>> QCal(method_name='hdx', n_validation=3)
    '''

    # Maps method_name -> quantifier class + whether a pre-trained
    # classifier (clf) is needed to build the estimates. The extraction
    # itself is now handled by the single, generic
    # extract_estimates_from_train_iso (see utils_qcal.extract_estimates),
    # so no per-method extraction function is listed here anymore.
    _REGISTRY = {
        'cc':  {'quantifier': CC,     'uses_clf': True},
        'acc': {'quantifier': AC,     'uses_clf': True},
        'pac': {'quantifier': PAC,    'uses_clf': True},
        'dys': {'quantifier': DyS,    'uses_clf': True},
        'ms':  {'quantifier': MS,     'uses_clf': True},
        'emq': {'quantifier': EMQ,    'uses_clf': True},
        'fm':  {'quantifier': FM,     'uses_clf': True},
        'kde': {'quantifier': KDEyML, 'uses_clf': True},
        'hdx': {'quantifier': HDx,    'uses_clf': False},
    }

    def __init__(self, learner=None, *, method_name='cc', clf=None, n_validation=3,
                 n_prevalences=None, name_data=None, method=None, calibrator=False,
                 plot=False, plot_dir=None):
        if method_name not in self._REGISTRY:
            raise ValueError(
                f"method_name={method_name!r} unknown. "
                f"Valid options: {sorted(self._REGISTRY)}"
            )
        self.learner = learner
        self.method_name = method_name
        self.clf = clf
        self.n_validation = n_validation
        self.n_prevalences = n_prevalences
        self.name_data = name_data
        self.method = method           # calibration method name forwarded to pre_treined_model (e.g. 'BCTS')
        self.calibrator = calibrator  # bool flag forwarded to pre_treined_model's `calibration` arg
        self.calibrator = calibrator
        self.plot = plot              # if True, fit() also builds plot_estimates()/plot_calibration()
        self.plot_dir = plot_dir      # directory where fit() auto-saves the two diagnostic figures (if plot=True)
        self.regressor = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip"
        )
        self.quantifier = None
        # populated by fit(); used by plot_estimates()/plot_calibration()
        self.estimates_all_ = None
        self.true_prevalences_all_ = None
        # populated by fit() only when plot=True
        self.fig_estimates_ = None
        self.fig_calibration_ = None
        # populated by fit() only when plot=True; records where each
        # figure actually got saved (or None if no path was resolved)
        self.estimates_plot_path_ = None
        self.calibration_plot_path_ = None

    def fit(self, X_train, y_train, name_data=None,
            estimates_plot_path=None, calibration_plot_path=None, verbose_plot=True):
        '''
        estimates_plot_path / calibration_plot_path : str or Path, optional
            Only used when self.plot=True. If given, the corresponding
            diagnostic figure is saved to that exact path (parent
            directories are created automatically), taking precedence over
            self.plot_dir. If left as None and self.plot_dir was set in the
            constructor, the paths are built automatically as
            '{plot_dir}/{name_data}_{method_name}_estimates.png' and
            '{plot_dir}/{name_data}_{method_name}_calibration.png'. If
            neither is given, the figures are still built and stored in
            self.fig_estimates_ / self.fig_calibration_, but nothing is
            written to disk.
        verbose_plot : bool, default=True
            If True (and self.plot=True and a save path was resolved),
            prints the path each figure was saved to — handy when running
            many datasets/methods in a loop and you want to confirm where
            each experiment's plots landed.
        '''
        # if nothing is passed here, use the name saved in the constructor
        if name_data is None:
            name_data = self.name_data

        cfg = self._REGISTRY[self.method_name]
        quantifier_cls = cfg['quantifier']
        uses_clf = cfg['uses_clf']

        true_prevalences_all = []
        estimates_all = []
        for i in range(self.n_validation):
            # Creating validation set
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.3, stratify=y_train
            )
            # Generating subgroups. `n_prevalences` is only forwarded when
            # explicitly set: UPP_protocol_mlquantify's own default (1100) is
            # int, but mlquantify's UPP rejects an explicit n_prevalences=None
            # (raises ValueError), so self.n_prevalences staying at its own
            # default of None must NOT be passed through.
            upp_kwargs = {} if self.n_prevalences is None else {'n_prevalences': self.n_prevalences}
            subgroups_train = UPP_protocol_mlquantify(X_val, y_val, **upp_kwargs)

            model_clf = None
            if uses_clf:
                model_clf = pre_treined_model(
                    X_tr, y_tr, clf=self.clf, method=self.method, calibration=self.calibrator
                )
            # Single generic extraction function for every method_name;
            # it internally builds+fits `quantifier_cls` and reads off the
            # class-1 prevalence estimate for each subgroup.
            estimates, true_prevalences = extract_estimates_from_train_iso(
                X_tr, y_tr, subgroups_train,
                quantifier_cls=quantifier_cls, model_clf=model_clf
            )

            # Storing the results of each validation
            estimates_all.extend(estimates)
            # Storing the true proportions of each validation
            true_prevalences_all.extend(true_prevalences)

        # Instantiating and fitting the final quantifier on the full training
        # set. self.learner is ALREADY a trained classifier (it comes from
        # pre_treined_model, calibrated or not), so fit_classifier=False: KDEyML
        # must use it as-is instead of retraining it. This also keeps the final
        # quantifier consistent with (a) the quantifiers built inside the
        # validation loop, which also get an already-trained model_clf, and
        # (b) the KDEyML baseline in the experiment, which is likewise built
        # with fit_classifier=False.
        self.quantifier = _build_and_fit_quantifier(
            quantifier_cls, X_train, y_train, model_clf=self.learner, fit_classifier=False
        )
        self.regressor.fit(estimates_all, true_prevalences_all)

        # kept around so plot_estimates()/plot_calibration() can be called
        # after fit() without needing to re-run the validation loop
        self.estimates_all_ = estimates_all
        self.true_prevalences_all_ = true_prevalences_all

        if self.plot:
            # Explicit paths passed to fit() always win. Otherwise, if
            # plot_dir was set in the constructor, build the paths
            # automatically from it + name_data + method_name.
            if estimates_plot_path is None and self.plot_dir is not None:
                base = name_data or "data"
                estimates_plot_path = os.path.join(
                    str(self.plot_dir), f"{base}_{self.method_name}_estimates.pdf"
                )
            if calibration_plot_path is None and self.plot_dir is not None:
                base = name_data or "data"
                calibration_plot_path = os.path.join(
                    str(self.plot_dir), f"{base}_{self.method_name}_calibration.pdf"
                )

            self.fig_estimates_, _ = self.plot_estimates(
                dataset_name=name_data, save_path=estimates_plot_path
            )
            self.fig_calibration_, _ = self.plot_calibration(
                dataset_name=name_data, save_path=calibration_plot_path
            )

            # kept around so you can check afterwards where each
            # experiment's figures were written (or None if not saved)
            self.estimates_plot_path_ = estimates_plot_path
            self.calibration_plot_path_ = calibration_plot_path

            if verbose_plot:
                if estimates_plot_path is not None:
                    print(f"[QCal] estimates plot saved to: {estimates_plot_path}")
                if calibration_plot_path is not None:
                    print(f"[QCal] calibration plot saved to: {calibration_plot_path}")

        return self

    def predict(self, X_test):
        predict_quantifier = extract_estimates_from_test(X_test, self.quantifier)

        class1 = predict_quantifier[:, 1]
        prev_class1 = self.regressor.predict(class1)[0]
        prev_class0 = 1.0 - prev_class1

        return np.array([prev_class0, prev_class1])

    def plot_estimates(self, dataset_name=None, save_path=None):
        '''
        Scatter plot of the raw quantifier estimates (before isotonic
        correction) against the true prevalences, collected during fit()'s
        n_validation loop. Generalizes the old per-method plot_cc_vs_real to
        any `method_name`.

        save_path : str or Path, optional
            If given, saves the figure to this path (parent directories are
            created automatically), e.g. 'plots/dataset1_cc_estimates.png'.

        Must be called after fit(). Returns (fig, ax).
        '''
        if self.estimates_all_ is None:
            raise RuntimeError(
                "No validation estimates found — call fit() before plot_estimates()."
            )
        return plot_estimates_vs_real(
            self.true_prevalences_all_, self.estimates_all_,
            method_name=self.method_name, dataset_name=dataset_name or self.name_data,
            save_path=save_path
        )

    def plot_calibration(self, dataset_name=None, save_path=None):
        '''
        IN-SAMPLE fit diagnostic for the isotonic-regression correction: raw
        estimate vs. true prevalence (the pairs used to fit self.regressor),
        the fitted isotonic curve, and the ideal diagonal.

        NOTE: the pairs plotted here are exactly the ones self.regressor was
        FITTED on (they come from fit()'s validation loop). This figure is
        therefore a diagnostic of the *learned correction* — its shape, and
        whether it degenerated into a near-constant step function — and NOT
        evidence that the correction generalizes. For that, use
        plot_test_calibration(), which evaluates the correction on held-out
        test subgroups.

        save_path : str or Path, optional
            If given, saves the figure to this path (parent directories are
            created automatically), e.g. 'plots/dataset1_cc_calibration.png'.

        Must be called after fit(). Returns (fig, ax).
        '''
        if self.estimates_all_ is None:
            raise RuntimeError(
                "No validation estimates found — call fit() before plot_calibration()."
            )
        return plot_isotonic_calibration(
            self.true_prevalences_all_, self.estimates_all_, self.regressor,
            method_name=self.method_name, dataset_name=dataset_name or self.name_data,
            save_path=save_path
        )

    def plot_test_calibration(self, X_test, y_test, dataset_name=None, save_path=None,
                              return_data=False, with_curve=False, **plot_kwargs):
        '''
        OUT-OF-SAMPLE calibration plot: evaluates the QCal correction on
        subgroups drawn from the TEST set.

        Unlike plot_calibration() (which re-plots the pairs the isotonic
        regressor was fitted on), this method:
          1. builds fresh subgroups from (X_test, y_test) via the UPP protocol;
          2. reads the RAW class-1 estimate of self.quantifier for each
             subgroup — the quantifier is used as-is, already fitted on the
             full training set, and is NOT refitted here;
          3. passes those raw estimates through self.regressor to obtain the
             CORRECTED estimates;
          4. plots the result.

        Neither the quantifier nor the regressor ever saw these subgroups, so
        the resulting figure is a legitimate generalization check.

        Parameters
        ----------
        X_test, y_test : the held-out test set (never passed to fit()).
        dataset_name : optional name shown in the title.
        save_path : optional path to save the figure to.
        with_curve : chooses which of the two out-of-sample views to draw.
            - False (default): axes are (true prevalence, estimate). Shows the
              RAW cloud and the CORRECTED cloud against the diagonal, with the
              test MAE of each in the legend. Best for showing that the
              correction moves the estimates toward the truth. The isotonic
              curve cannot be drawn on these axes (it is a function of the raw
              estimate, not of the true prevalence — the corrected points ARE
              the curve applied).
            - True: axes are (raw estimate, true prevalence), i.e. the same
              axes as plot_calibration(), so the fitted isotonic CURVE is drawn
              — but scattered against the held-out test pairs instead of the
              validation pairs it was fitted on. Use this when you want the
              familiar calibration-curve look while still being out-of-sample.
        return_data : if True, also returns (raw_estimates, corrected_estimates,
            true_prevalences) — handy for logging the test MAE of raw vs.
            corrected into your results CSV.
        **plot_kwargs : forwarded to the underlying plotting function
            (title_fontsize, label_fontsize, show_title, tight_bbox, dpi, ...).

        Must be called after fit(). Returns (fig, ax) by default.
        '''
        if self.quantifier is None:
            raise RuntimeError(
                "No fitted quantifier found — call fit() before plot_test_calibration()."
            )

        # Fresh subgroups from the test set (same protocol used in fit()).
        subgroups_test = UPP_protocol_mlquantify(X_test, y_test)

        # Raw estimates from the ALREADY-fitted quantifier (no refitting).
        raw_estimates, true_prevalences = extract_estimates_from_test_subgroups(
            subgroups_test, self.quantifier
        )

        # Corrected estimates: raw values passed through the fitted regressor.
        corrected_estimates = self.regressor.predict(np.asarray(raw_estimates, dtype=float))

        if with_curve:
            # (raw estimate, true prevalence) axes -> the isotonic curve fits here
            fig, ax = plot_test_isotonic_curve(
                true_prevalences, raw_estimates, self.regressor,
                method_name=self.method_name, dataset_name=dataset_name or self.name_data,
                save_path=save_path, **plot_kwargs
            )
        else:
            # (true prevalence, estimate) axes -> raw cloud vs. corrected cloud
            fig, ax = plot_test_calibration(
                true_prevalences, raw_estimates, corrected_estimates,
                method_name=self.method_name, dataset_name=dataset_name or self.name_data,
                save_path=save_path, **plot_kwargs
            )

        if return_data:
            return fig, ax, raw_estimates, corrected_estimates, true_prevalences
        return fig, ax