from pathlib import Path
import textwrap
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Global font sizes + deterministic layout for every figure in this module.
#
# The figures are saved on a fixed canvas and then shrunk to ~0.48 \textwidth
# in a 2x2 LaTeX subfigure grid, so in-figure text has to be LARGE to stay
# legible in print. Edit these numbers HERE and every plot follows; or
# override per call (e.g. plot_test_calibration(..., legend_fontsize=20)); or
# from a script:
#
#     import qcal_plots
#     qcal_plots.FONT["title"] = 30
#
# Sizes resolve at call time (defaults are None), so assigning to FONT after
# import works.
# ---------------------------------------------------------------------------
FONT = {
    "title": 26,      # bold title (usually OFF in a grid; the panel gets a \subcaption instead)
    "label": 26,      # axis labels
    "tick": 22,       # tick numbers
    "legend": 22,     # legend text (OUTSIDE the axes, so it can be large)
    "wrap": 30,       # chars per title line (lower it if a title still clips)
}

# Where to put the legend relative to the plotting area. It is ALWAYS outside
# the axes (never on top of the data): "below" places it under the plot,
# "right" places it to the side. "below" keeps every panel the same width and
# is the better default for a 2x2 grid.
LEGEND_LOC = "below"


def _fs(value, key):
    """Per-call font size if given, otherwise the module-wide FONT default."""
    return FONT[key] if value is None else value


def _wrap_title(title, width=None):
    '''
    Wraps a long title over several lines.

    Needed because these figures are saved on a fixed canvas (bbox_inches=None,
    so that multi-panel LaTeX grids line up). On a fixed canvas a long bold
    title simply overflows the figure width and gets CLIPPED at both ends —
    e.g. "Isotonic Regression Calibration of CC on the balance.3 dataset"
    rendering as "onic Regression Calibration of CC on the balance.3 dat".
    `constrained_layout` reflows the vertical space to fit the title, but it
    does not break the line for you. Wrapping does.
    '''
    width = FONT["wrap"] if width is None else width
    return "\n".join(textwrap.wrap(title, width=width)) if title else title


def _figsize(legend_loc):
    '''
    Canvas size chosen so that, after `constrained_layout` reserves room for
    the OUTSIDE legend, the square plotting area still fills the panel without
    big empty margins. Same for every figure of a given legend placement, so
    all panels of a 2x2 grid render at identical size and line up.
    '''
    if legend_loc == "right":
        return (12.5, 8.0)   # extra width on the right for the legend column
    return (8.0, 9.5)        # extra height at the bottom for the legend row


def _add_outside_legend(fig, handles, labels, legend_loc, fontsize):
    '''
    Puts the legend OUTSIDE the plotting area and returns it.

    It uses `fig.legend(loc="outside ...")`, which — unlike an axes legend
    anchored with `bbox_to_anchor` — is genuinely RESERVED by
    `constrained_layout`: the axes are shrunk to make room and the legend is
    placed inside the fixed canvas. That is what prevents the legend from
    being clipped at the canvas edge when saving with `bbox_inches=None`, while
    still keeping every panel the same size for a 2x2 grid.
    '''
    if legend_loc == "right":
        leg = fig.legend(
            handles, labels,
            loc="outside center right",
            frameon=True, framealpha=0.9, fontsize=fontsize,
            markerscale=2, handlelength=1.6,
        )
    else:  # "below"
        leg = fig.legend(
            handles, labels,
            loc="outside lower center", ncol=1,
            frameon=True, framealpha=0.9, fontsize=fontsize,
            markerscale=2, handlelength=1.6,
        )
    
    # scatter handles are drawn with low alpha; make them opaque in the legend
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return leg


def _save(fig, save_path, dpi, tight_bbox, leg):
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if tight_bbox:
        # Crop to drawn content (the outside legend is part of the figure and
        # is included automatically). NOTE: with tight bbox the output size
        # depends on the legend/title extent, which breaks strict grid
        # alignment; prefer the default (tight_bbox=False) for panels that
        # must line up.
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    bbox_extra_artists=(leg,))
    else:
        fig.savefig(save_path, dpi=dpi, bbox_inches=None)


def plot_estimates_vs_real(true_prevalences, estimates, method_name=None, dataset_name=None,
                            save_path=None, dpi=150, title_fontsize=None, label_fontsize=None,
                            tick_fontsize=None, legend_fontsize=None, show_title=False,
                            tight_bbox=False, legend_loc=None):
    '''
    Scatter plot of the raw quantifier estimates (before isotonic correction)
    against the true prevalences, collected over the n_validation splits.

    Generalization of the original `plot_cc_vs_real`: works for any
    `method_name` ('cc', 'acc', 'pac', 'dys', 'ms', 'emq', 'fm', 'kde', 'hdx'),
    not just CC.

    Parameters
    ----------
    - true_prevalences: list/array of true class-1 prevalences (one per
      validation subgroup, across all n_validation splits);
    - estimates: list/array of raw estimates from the base quantifier for the
      same subgroups (same order/length as true_prevalences);
    - method_name: name of the base quantifier (only used for the legend and
      axis label, e.g. 'cc', 'kde');
    - dataset_name: optional dataset name, shown in the title;
    - save_path: optional path (str or Path) to save the figure to. Parent
      directories are created automatically. If None, the figure is not saved
      to disk — only returned.
    - dpi: resolution used when saving (ignored if save_path is None).
    - title_fontsize, label_fontsize, tick_fontsize, legend_fontsize: font
      sizes; defaults come from the module-wide FONT dict and are sized to
      survive shrinking into a paper column.
    - show_title: if False (default), no in-figure title is drawn (use a LaTeX
      \\subcaption for the panel instead — recommended in a grid).
    - tight_bbox: if False (default), the saved file keeps exactly the canvas
      size given by `figsize`, so every figure has the SAME dimensions and the
      panels line up in a subfigure grid. If True, the file is cropped to the
      drawn content (the outside legend is still included), which is handy for
      a standalone figure but makes the size depend on the legend/title extent.
    - legend_loc: "below" (default) or "right". The legend is always OUTSIDE
      the plotting area, never on top of the data, and is reserved inside the
      canvas so it is not clipped.

    Returns
    ----
    - fig, ax: the matplotlib Figure and Axes objects.
    '''
    x = np.asarray(true_prevalences)
    y = np.asarray(estimates)

    legend_loc = LEGEND_LOC if legend_loc is None else legend_loc
    method_label = method_name.upper() if method_name else "quantifier"

    fig, ax = plt.subplots(figsize=_figsize(legend_loc), constrained_layout={"h_pad": 0.2})

    ax.scatter(x, y, s=8, alpha=0.25, edgecolor="none",
               color="#2c7fb8", rasterized=True, label=f"Predictions ({method_label})")
    ax.plot([0, 1], [0, 1], "--", color="0.3", lw=1.5, label="Ideal")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Positive class prevalence", fontsize=_fs(label_fontsize, "label"))
    ax.set_ylabel("Estimated prevalence", fontsize=_fs(label_fontsize, "label"))

    if show_title:
        if dataset_name:
            title = f"Performance of {method_label} on the {dataset_name} dataset"
        else:
            title = f"Performance of {method_label}"
        ax.set_title(_wrap_title(title), fontsize=_fs(title_fontsize, "title"), fontweight="bold")

    ax.tick_params(axis="both", labelsize=_fs(tick_fontsize, "tick"))
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    leg = _add_outside_legend(fig, handles, labels, legend_loc, _fs(legend_fontsize, "legend"))

    _save(fig, save_path, dpi, tight_bbox, leg)
    return fig, ax


def plot_isotonic_calibration(true_prevalences, estimates, regressor,
                               method_name=None, dataset_name=None, n_grid=300,
                               save_path=None, dpi=150, title_fontsize=None, label_fontsize=None,
                               tick_fontsize=None, legend_fontsize=None, show_title=False,
                               tight_bbox=False, legend_loc=None):
    '''
    Calibration plot for the QCal isotonic-regression correction: shows the
    raw quantifier estimates against the true prevalences (the pairs used to
    fit `regressor`), together with the fitted isotonic curve (raw estimate
    -> corrected estimate) and the ideal diagonal.

    Parameters
    ----------
    - true_prevalences: true class-1 prevalences (the regression targets);
    - estimates: raw quantifier estimates (the regression inputs), same
      order/length as true_prevalences;
    - regressor: the fitted `sklearn.isotonic.IsotonicRegression` instance;
    - method_name: name of the base quantifier (labeling only);
    - dataset_name: optional dataset name, shown in the title;
    - n_grid: number of points used to trace the fitted isotonic curve;
    - save_path, dpi, *_fontsize, show_title, tight_bbox, legend_loc: as in
      `plot_estimates_vs_real`.

    Returns
    ----
    - fig, ax: the matplotlib Figure and Axes objects.
    '''
    x_raw = np.asarray(estimates)
    y_true = np.asarray(true_prevalences)

    legend_loc = LEGEND_LOC if legend_loc is None else legend_loc
    method_label = method_name.upper() if method_name else "quantifier"

    grid = np.linspace(0.0, 1.0, n_grid)
    corrected = regressor.predict(grid)

    fig, ax = plt.subplots(figsize=_figsize(legend_loc), constrained_layout={"h_pad": 0.2})

    ax.scatter(x_raw, y_true, s=8, alpha=0.25, edgecolor="none",
               color="#2c7fb8", rasterized=True,
               label="Pairs (raw estimate, true prevalence)")
    ax.plot(grid, corrected, color="#d95f02", lw=2,
            label="Fitted correction (Isotonic Regression)")
    ax.plot([0, 1], [0, 1], "--", color="0.3", lw=1.5,
            label="Ideal (no correction needed)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Raw estimate ({method_label})", fontsize=_fs(label_fontsize, "label"))
    ax.set_ylabel("True prevalence / corrected value", fontsize=_fs(label_fontsize, "label"))

    if show_title:
        if dataset_name:
            title = f"Isotonic Regression Calibration of {method_label} on the {dataset_name} dataset"
        else:
            title = f"Isotonic Regression Calibration of {method_label}"
        ax.set_title(_wrap_title(title), fontsize=_fs(title_fontsize, "title"), fontweight="bold")

    ax.tick_params(axis="both", labelsize=_fs(tick_fontsize, "tick"))
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    leg = _add_outside_legend(fig, handles, labels, legend_loc, _fs(legend_fontsize, "legend"))

    _save(fig, save_path, dpi, tight_bbox, leg)
    return fig, ax


def plot_test_calibration(true_prevalences, raw_estimates, corrected_estimates,
                          method_name=None, dataset_name=None,
                          save_path=None, dpi=150, title_fontsize=None, label_fontsize=None,
                          tick_fontsize=None, legend_fontsize=None, show_title=False,
                          tight_bbox=False, show_mae=True, legend_loc=None):
    '''
    OUT-OF-SAMPLE calibration plot: the effect of the QCal isotonic correction
    evaluated on subgroups drawn from the TEST set.

    Why this exists (and how it differs from `plot_isotonic_calibration`)
    --------------------------------------------------------------------
    `plot_isotonic_calibration` plots the very pairs that were used to FIT the
    isotonic regressor, which is a fit diagnostic, not evidence of
    generalization. This function instead uses held-out TEST subgroups (via
    UPP) and plots, for each:
      - the RAW estimate vs. the true prevalence, and
      - the CORRECTED estimate vs. the true prevalence,
    against the ideal diagonal. If QCal works, the orange (corrected) cloud
    sits closer to the diagonal than the blue (raw) cloud.

    Parameters
    ----------
    - true_prevalences: true class-1 prevalence of each TEST subgroup;
    - raw_estimates: base quantifier's class-1 estimate for those subgroups;
    - corrected_estimates: the same estimates after `regressor.predict(...)`;
    - method_name, dataset_name, save_path, dpi, *_fontsize, show_title,
      tight_bbox, legend_loc: as in the other plotting functions;
    - show_mae: if True, the MAE of the raw and corrected estimates is appended
      to the legend labels.

    Returns
    ----
    - fig, ax: the matplotlib Figure and Axes objects.
    '''
    y_true = np.asarray(true_prevalences, dtype=float)
    y_raw = np.asarray(raw_estimates, dtype=float)
    y_cor = np.asarray(corrected_estimates, dtype=float)

    legend_loc = LEGEND_LOC if legend_loc is None else legend_loc
    method_label = method_name.upper() if method_name else "quantifier"

    label_raw = f"Raw {method_label}"
    label_cor = f"Calibrated {method_label} (QCal)"
    if show_mae:
        mae_raw = float(np.mean(np.abs(y_raw - y_true)))
        mae_cor = float(np.mean(np.abs(y_cor - y_true)))
        label_raw += f" — MAE = {mae_raw:.4f}"
        label_cor += f" — MAE = {mae_cor:.4f}"

    fig, ax = plt.subplots(figsize=_figsize(legend_loc), constrained_layout={"h_pad": 0.2})

    ax.scatter(y_true, y_raw, s=10, alpha=0.30, edgecolor="none",
               color="#2c7fb8", rasterized=True, label=label_raw)
    ax.scatter(y_true, y_cor, s=10, alpha=0.30, edgecolor="none",
               color="#d95f02", rasterized=True, label=label_cor)
    ax.plot([0, 1], [0, 1], "--", color="0.3", lw=1.5, label="Ideal")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Positive class prevalence (test)", fontsize=_fs(label_fontsize, "label"))
    ax.set_ylabel("Estimated prevalence", fontsize=_fs(label_fontsize, "label"))

    # if show_title:
    #     if dataset_name:
    #         title = f"Effect of QCal on {method_label} — {dataset_name} (test set)"
    #     else:
    #         title = f"Effect of QCal on {method_label} (test set)"
    #     ax.set_title(_wrap_title(title), fontsize=_fs(title_fontsize, "title"), fontweight="bold")

    ax.tick_params(axis="both", labelsize=_fs(tick_fontsize, "tick"))
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    leg = _add_outside_legend(fig, handles, labels, legend_loc, _fs(legend_fontsize, "legend"))

    _save(fig, save_path, dpi, tight_bbox, leg)
    return fig, ax


def plot_test_isotonic_curve(true_prevalences, raw_estimates, regressor,
                             method_name=None, dataset_name=None, n_grid=300,
                             save_path=None, dpi=150, title_fontsize=None, label_fontsize=None,
                             tick_fontsize=None, legend_fontsize=None, show_title=False,
                             tight_bbox=False, legend_loc=None):
    '''
    Same axes and same look as `plot_isotonic_calibration` — including the
    fitted isotonic curve — but the scatter points come from the TEST set.
    Reading it: if the orange curve (learned on validation) passes through the
    middle of the blue test cloud, the correction transferred to unseen data.

    Parameters
    ----------
    - true_prevalences: true class-1 prevalence of each TEST subgroup;
    - raw_estimates: raw class-1 estimate of the (already fitted) base
      quantifier for those same TEST subgroups;
    - regressor: the fitted IsotonicRegression (fitted on the VALIDATION
      pairs inside QCal.fit(), not on these test pairs);
    - everything else: as in the other plotting functions in this module.

    Returns
    ----
    - fig, ax: the matplotlib Figure and Axes objects.
    '''
    x_raw = np.asarray(raw_estimates, dtype=float)
    y_true = np.asarray(true_prevalences, dtype=float)

    legend_loc = LEGEND_LOC if legend_loc is None else legend_loc
    method_label = method_name.upper() if method_name else "quantifier"

    grid = np.linspace(0.0, 1.0, n_grid)
    corrected = regressor.predict(grid)

    fig, ax = plt.subplots(figsize=_figsize(legend_loc), constrained_layout={"h_pad": 0.2})

    ax.scatter(x_raw, y_true, s=8, alpha=0.25, edgecolor="none",
               color="#2c7fb8", rasterized=True,
               label="Test pairs (raw estimate, true prevalence)")
    ax.plot(grid, corrected, color="#d95f02", lw=2,
            label="Fitted correction (learned on validation)")
    ax.plot([0, 1], [0, 1], "--", color="0.3", lw=1.5,
            label="Ideal (no correction needed)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Raw estimate ({method_label})", fontsize=_fs(label_fontsize, "label"))
    ax.set_ylabel("Corrected value", fontsize=_fs(label_fontsize, "label"))

    if show_title:
        if dataset_name:
            title = f"QCal correction of {method_label} on {dataset_name} (test set)"
        else:
            title = f"QCal correction of {method_label} (test set)"
        ax.set_title(_wrap_title(title), fontsize=_fs(title_fontsize, "title"), fontweight="bold")

    ax.tick_params(axis="both", labelsize=_fs(tick_fontsize, "tick"))
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    leg = _add_outside_legend(fig, handles, labels, legend_loc, _fs(legend_fontsize, "legend"))

    _save(fig, save_path, dpi, tight_bbox, leg)
    return fig, ax