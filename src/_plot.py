from copy import copy

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.patches import Ellipse, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.calibration import calibration_curve as sklearn_calibration_curve
from statsmodels.stats.proportion import proportion_confint

from src.grouping_loss import calibration_curve


def set_latex_font(math=True, normal=True, extra_preamble=[]):
    if math:
        plt.rcParams["mathtext.fontset"] = "stix"
    else:
        plt.rcParams["mathtext.fontset"] = plt.rcParamsDefault["mathtext.fontset"]

    if normal:
        plt.rcParams["font.family"] = "STIXGeneral"
    else:
        plt.rcParams["font.family"] = plt.rcParamsDefault["font.family"]

    usetex = mpl.checkdep_usetex(True)
    plt.rc("text", usetex=usetex)
    default_preamble = [
        r"\usepackage{amsfonts}",
    ]
    preamble = "".join(default_preamble + extra_preamble)
    plt.rc("text.latex", preamble=preamble)


def separating_line2D(X, beta, beta0):
    assert beta.shape[0] == 2
    return -1 / beta[1] * (X * beta[0] + beta0)


def plot_covariance2D(
    mean,
    cov,
    ax,
    n_std=3.0,
    facecolor="none",
    edgecolor="black",
    linestyle="--",
    label=r"$\Sigma$",
    **kwargs,
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    assert cov.ndim == 2
    assert mean.shape == (2,)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=linestyle,
        label=label,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_score_vs_probas2(
    y_scores,
    y_labels,
    y_true_probas,
    n_bins=15,
    samples_with_mv=None,
    legend_loc="best",
    ncol=3,
    lim_margin=0.15,
    max_samples=None,
    grid_space=0.2,
    height=6,
    plot_first_last_bins=True,
):
    """Plot confidence score outputted by a classifier versus the
    true probability of the samples and color points according to their label.
    """
    set_latex_font()

    y_scores = np.array(y_scores)
    y_true_probas = np.array(y_true_probas)
    y_labels = np.array(y_labels)

    prob_bins, mean_bins = sklearn_calibration_curve(y_labels, y_scores, n_bins=n_bins)

    if max_samples is not None:
        y_scores = y_scores[:max_samples]
        y_true_probas = y_true_probas[:max_samples]
        y_labels = y_labels[:max_samples]

    _y_labels = np.full(y_labels.shape, "Negative")
    _y_labels[y_labels == 1] = "Positive"
    hue_order = ["Negative", "Positive"]
    df = pd.DataFrame(
        {
            "y_scores": y_scores,
            "y_true_probas": y_true_probas,
            "y_labels": y_labels,
            "_y_labels": _y_labels,
        }
    )
    g = sns.JointGrid(
        data=df,
        x="y_scores",
        y="y_true_probas",
        hue="_y_labels",
        ratio=10,
        space=grid_space,
        height=height,
        hue_order=hue_order,
    )

    def scatter_with_mv(x, y, hue, missing):
        ax = plt.gca()
        style = pd.Series("Complete", index=x[~missing].index)
        sns.scatterplot(
            x=x[~missing],
            y=y[~missing],
            hue=hue,
            alpha=1,
            ax=ax,
            style=style,
            style_order=["Complete", "Missing"],
        )
        sns.scatterplot(
            x=x[missing],
            y=y[missing],
            hue=hue,
            alpha=1,
            ax=ax,
            legend=False,
            palette="pastel",
            marker="X",
        )

    if samples_with_mv is not None:
        samples_with_mv = np.array(samples_with_mv)
        g.plot_joint(scatter_with_mv, missing=samples_with_mv)
    else:
        g.plot_joint(sns.scatterplot, s=15)

    bins = np.linspace(0, 1, n_bins + 1)

    def histplot_with_size(x, vertical, hue):
        color_hue = np.ones_like(x)
        if vertical:
            x, y = None, x
        else:
            x, y = x, None
        sns.histplot(x=x, y=y, bins=list(bins), legend=False, color="silver")

    g.plot_marginals(histplot_with_size)
    g.set_axis_labels(xlabel="Confidence score", ylabel="True probability")
    ax = g.figure.axes[0]
    ax.legend(title="Label")

    if not plot_first_last_bins:
        bins = bins[1:-1]
    for x in bins:
        ax.axvline(x, lw=0.5, ls="--", color="grey", zorder=-1)

    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="black")
    ax.plot(
        mean_bins, prob_bins, marker=".", markersize=5, color="black"
    )

    ax.legend(loc=legend_loc, ncol=ncol)
    if lim_margin is not None:
        ax.set_xlim((-lim_margin, 1 + lim_margin))
        ax.set_ylim((-lim_margin, 1 + lim_margin))
    return ax.figure


def insert_nan_at_discontinuities(X, Y, min_gap=0.1):
    """Look at discontinuities in Y and add nan at their position for
    discontinuous plotting."""
    X, Y = X.copy(), Y.copy()
    pos = np.where(np.abs(np.diff(Y)) > min_gap)[0] + 1
    if len(pos) > 0:
        X = np.insert(X, pos, np.nan)
        Y = np.insert(Y, pos, np.nan)
    return X, Y


def plot_ffstar_1d(
    f,
    f_star,
    p,
    x_min,
    x_max,
    disc_gap=0.1,
    figsize=(2, 2),
    bbox_to_anchor=(1, 0),
    loc="lower right",
    lw=1,
    frameon=True,
):
    set_latex_font()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 10], hspace=0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    plt.subplots_adjust(hspace=0.075)
    XX = np.linspace(x_min, x_max, len(p))
    ax1.plot(XX, p, label="$\\mathbb{P}_X$", lw=0.5)
    ax1.fill_between(XX, p, alpha=0.2)
    ax1.get_yaxis().set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.set_xticklabels([])
    ylim = ax1.get_ylim()
    ax1.set_ylim((0, ylim[1]))

    ax2.axhline(0.5, color="black", lw=0.5)
    XX = np.linspace(x_min, x_max, len(f))
    XX, YY = insert_nan_at_discontinuities(XX, f, min_gap=disc_gap)
    ax2.plot(XX, YY, label="$S(X)$", color="black", lw=lw)
    XX = np.linspace(x_min, x_max, len(f_star))
    YY = f_star
    ax2.plot(XX, YY, label="$Q(X)$", color="tab:red", ls="-", lw=lw)
    ax2.set_xlabel("$X$")
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(["0", "$\\frac{1}{2}$", "1"])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles=h2 + h1,
        labels=l2 + l1,
        bbox_to_anchor=bbox_to_anchor,
        loc=loc,
        frameon=frameon,
    )
    ax2.spines["top"].set_visible(False)

    return fig


def plot_ffstar_2d_v2(
    f,
    phi,
    psi=None,
    delta=None,
    delta_max=None,
    w=None,
    w_orth=None,
    mean=None,
    cov=None,
    w_learned=None,
    trim=False,
    figsize=(4.5, 6),
):
    set_latex_font()

    plot_y_label = True
    fontsize_xcal = 9
    fontsize = 12
    h = 100
    x_min = -2
    x_max = 2
    y_min = -2
    y_max = 2
    XX0, YY0 = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))

    cm = plt.cm.RdBu_r

    if trim:
        fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    else:
        fig1 = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, figure=fig1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

    if w is not None:
        ax1.annotate(
            "",
            xy=(w[0], w[1]),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                shrinkB=0,
                patchB=None,
                patchA=None,
                shrinkA=0,
                color="black",
            ),
        )
        ax1.text(w[0] / 2, w[1] / 2, r"$\omega$", va="top", ha="left")
        X1 = np.linspace(x_min, x_max, 2)
        X2 = separating_line2D(X1, w, 0)
        ax1.plot(
            X1, X2, color="black", linestyle="-", lw=0.5, label=r"$S = \frac{1}{2}$"
        )

    if mean is not None and cov is not None:
        plot_covariance2D(
            mean,
            cov,
            ax1,
            n_std=0.7,
            edgecolor="black",
            lw=0.5,
            linestyle=":",
            label=r"$\Sigma$",
            zorder=3,
        )

    ax1.set_xlim((x_min, x_max))
    ax1.set_ylim((y_min, y_max))
    ax1.set_xticks([x_min, x_max])
    ax1.set_yticks([y_min, y_max])
    ax1.set_title(r"$S(X) = \varphi(\omega^TX)$")
    ax1.set_aspect("equal")
    ax1.legend(bbox_to_anchor=(0, 0), loc="lower left", ncol=2)

    d = 0.04
    ax1.annotate(
        "$X_1$",
        xy=(0.5, -d),
        xytext=(0.5, -d),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    d = 0.02
    ax1.annotate(
        "$X_2$",
        xy=(-d, 0.5),
        xytext=(-d, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=plt.rcParams["axes.labelsize"],
        rotation=90,
    )

    # Plot colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    Z1 = f(np.c_[XX0.ravel(), YY0.ravel()])
    Z1 = Z1.reshape(XX0.shape)
    crf = ax1.contourf(
        XX0, YY0, Z1, levels=np.linspace(0, 1, 101), cmap=cm, alpha=0.8, vmin=0, vmax=1
    )
    cbar = plt.colorbar(crf, cax=cax)
    cbar.ax.set_title(r"$S(X)$")
    cbar.ax.set_yticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])

    # Plot Xcal feature space in corner
    dxcal = 0.025
    x = dxcal
    y = 1 - dxcal
    ax1.annotate(
        "    ",
        xy=(x, y),
        xycoords="axes fraction",
        bbox=dict(boxstyle="square", ec="black", fc="white", alpha=1, linewidth=0.7),
        ha="left",
        va="top",
        fontsize=fontsize_xcal,
    )
    ax1.annotate(
        r"$\mathcal{X}~$",
        xy=(x, y),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=fontsize_xcal + 3,
    )

    if ax2 is not None:
        X1_min = -2
        X1_max = 2
        X1 = np.linspace(X1_min, X1_max, 500)
        Y1 = phi(X1)
        ax2.plot(X1, Y1, label=r"$\varphi(w^TX)$")
        ax2.set_xlabel(r"$w^TX$")
        ax2.set_xticks([X1_min, 0, X1_max])
        ax2.set_yticks([0, 0.5, 1])
        ax2.set_yticklabels(["0", "$\\frac{1}{2}$", "1"])
        ax2.axhline(0.5, color="black", lw=0.5)
        ax2.legend()

    if psi is None or delta is None or delta_max is None:
        return fig1

    if trim:
        fig2, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
        ax3 = None
    else:
        fig2 = plt.figure(figsize=(4.5, 8))
        gs = gridspec.GridSpec(3, 1, figure=fig2, height_ratios=[2.4, 1, 1], hspace=0.3)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])

    Z2 = delta(np.c_[XX0.ravel(), YY0.ravel()])
    Z2 = Z2.reshape(XX0.shape)
    Z2_max = np.max(Z2)
    Z2_min = np.min(Z2)
    Z2_lim = max(np.abs(Z2_max), np.abs(Z2_min))
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    crf = ax1.contourf(
        XX0,
        YY0,
        Z2,
        levels=np.linspace(-Z2_lim, Z2_lim, 101),
        cmap=cm,
        alpha=0.8,
        vmin=-Z2_lim,
        vmax=Z2_lim,
    )
    cbar = fig2.colorbar(crf, cax=cax)
    cbar.ax.set_title(r"$\Delta(X)$")
    cbar.set_ticks([-Z2_lim, 0, Z2_lim])
    cbar.ax.set_yticklabels([f"{-Z2_lim:.2g}", "0", f"{Z2_lim:.2g}"])

    if w is not None:
        ax1.annotate(
            "",
            xy=(w[0], w[1]),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                shrinkB=0,
                patchB=None,
                patchA=None,
                shrinkA=0,
                color="black",
            ),
        )
        ax1.text(
            w[0] / 2, w[1] / 2, r"$\omega$", va="top", ha="left", fontsize=fontsize
        )

    if w_orth is not None:
        ax1.annotate(
            "",
            xy=(w_orth[0], w_orth[1]),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                shrinkB=0,
                patchB=None,
                patchA=None,
                shrinkA=0,
                color="black",
            ),
        )
        ax1.text(
            w_orth[0] / 2,
            w_orth[1] / 2,
            r"$\omega_{\perp}$",
            va="top",
            ha="right",
            fontsize=fontsize,
        )
        X1 = np.linspace(x_min, x_max, 2)
        X2 = separating_line2D(X1, w_orth, 0)
        ax1.plot(
            X1,
            X2,
            color="black",
            linestyle="-",
            lw=0.5,
            label=r"$\omega_{\perp}^TX = 0$",
        )

    ax1.set_xlim((x_min, x_max))
    ax1.set_ylim((y_min, y_max))
    ax1.set_xticks([x_min, x_max])
    ax1.set_yticks([y_min, y_max])
    if not plot_y_label:
        ax1.set_yticklabels(["", ""])

    d = 0.04
    ax1.annotate(
        "$X_1$",
        xy=(0.5, -d),
        xytext=(0.5, -d),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.labelsize"],
    )

    if plot_y_label:
        d = 0.02
        ax1.annotate(
            "$X_2$",
            xy=(-d, 0.5),
            xytext=(-d, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=plt.rcParams["axes.labelsize"],
            rotation=90,
        )

    ax1.set_title(r"$\Delta(X) = \psi(\omega_{\perp}^TX)\Delta_{max}(X)$")
    ax1.set_aspect("equal")
    ax1.legend(bbox_to_anchor=(0, 0), loc="lower left")

    x = dxcal
    y = 1 - dxcal
    ax1.annotate(
        "    ",
        xy=(x, y),
        xycoords="axes fraction",
        bbox=dict(boxstyle="square", ec="black", fc="white", alpha=1, linewidth=0.7),
        ha="left",
        va="top",
        fontsize=fontsize_xcal,
    )
    ax1.annotate(
        r"$\mathcal{X}~$",
        xy=(x, y),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=fontsize_xcal + 3,
    )

    if ax2 is not None:
        X2_min = -10
        X2_max = 10
        X2 = np.linspace(X2_min, X2_max, 501)
        Y2 = psi(X2)
        ax2.plot(X2, Y2, label=r"$\psi(w_{\perp}^TX)$", color="tab:orange")
        ax2.set_xlabel(r"$w_{\perp}^TX$")
        ax2.set_xticks([X2_min, 0, X2_max])
        ax2.set_yticks([-1, 0, 1])
        ax2.axhline(0, color="black", lw=0.5)
        ax2.legend()

    if ax3 is not None:
        X2_min = -10
        X2_max = 10
        X2 = np.linspace(X2_min, X2_max, 501)
        Y2 = delta_max(X2)
        ax3.plot(X2, Y2, label=r"$\Delta_{max}(w^TX)$", color="tab:green")
        ax3.set_xlabel(r"$w^TX$")
        ax3.set_xticks([X2_min, 0, X2_max])
        ax3.set_yticks([0, Z2_lim])
        ax3.set_yticklabels(["0", f"{Z2_lim:.2g}"])
        ax3.legend()

    plot_y_label = False
    if trim:
        fig3, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    else:
        fig3 = plt.figure(figsize=(4.5, 6))
        gs = gridspec.GridSpec(2, 1, figure=fig3, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    Z3 = Z1 + Z2
    crf = ax1.contourf(
        XX0, YY0, Z3, levels=np.linspace(0, 1, 101), cmap=cm, alpha=0.8, vmin=0, vmax=1
    )
    cbar = fig3.colorbar(crf, cax=cax)
    cbar.ax.set_title(r"$Q(X)$")
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])

    if w is not None:
        ax1.annotate(
            "",
            xy=(w[0], w[1]),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                shrinkB=0,
                patchB=None,
                patchA=None,
                shrinkA=0,
                color="black",
            ),
        )
        ax1.text(
            w[0] / 2, w[1] / 2, r"$\omega$", va="top", ha="left", fontsize=fontsize
        )
        X1 = np.linspace(x_min, x_max, 2)
        X2 = separating_line2D(X1, w, 0)
        ax1.plot(
            X1, X2, color="black", linestyle="-", lw=0.5, label=r"$S(X) = \frac{1}{2}$"
        )

    if w_orth is not None:
        ax1.annotate(
            "",
            xy=(w_orth[0], w_orth[1]),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                shrinkB=0,
                patchB=None,
                patchA=None,
                shrinkA=0,
                color="black",
            ),
        )
        ax1.text(
            w_orth[0] / 2,
            w_orth[1] / 2,
            r"$\omega_{\perp}$",
            va="top",
            ha="right",
            fontsize=fontsize,
        )

    ax1.set_xlim((x_min, x_max))
    ax1.set_ylim((y_min, y_max))
    ax1.set_xticks([x_min, x_max])
    ax1.set_yticks([y_min, y_max])
    if not plot_y_label:
        ax1.set_yticklabels(["", ""])

    d = 0.04
    ax1.annotate(
        "$X_1$",
        xy=(0.5, -d),
        xytext=(0.5, -d),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    if plot_y_label:
        d = 0.02
        ax1.annotate(
            "$X_2$",
            xy=(-d, 0.5),
            xytext=(-d, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=plt.rcParams["axes.labelsize"],
            rotation=90,
        )
    ax1.set_title(r"$Q(X) = S(X) + \Delta(X)$")
    ax1.set_aspect("equal")
    ax1.legend(bbox_to_anchor=(0, 0), loc="lower left")

    x = dxcal
    y = 1 - dxcal
    ax1.annotate(
        "    ",
        xy=(x, y),
        xycoords="axes fraction",
        bbox=dict(boxstyle="square", ec="black", fc="white", alpha=1, linewidth=0.7),
        ha="left",
        va="top",
        fontsize=fontsize_xcal,
    )
    ax1.annotate(
        r"$\mathcal{X}~$",
        xy=(x, y),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=fontsize_xcal + 3,
    )

    if ax2 is not None:
        X1_min = -3
        X1_max = 3
        X2_min = -10
        X2_max = 10
        X1 = np.linspace(X1_min, X1_max, 8)
        X2 = np.linspace(X2_min, X2_max, 501)
        for i, x in enumerate(X1):
            y = phi(x)
            color = cm(y)
            ax2.plot([X2_min, X2_max], [y, y], ls=":", lw=0.75, color=color)
            _delta_max = delta_max(x)
            Y2 = y + np.multiply(psi(X2), _delta_max)
            ax2.plot(X2, Y2, color=color)
            if i == 0:
                ax2.plot([], [], color="black", ls=":", label=r"$f(X)$")
                ax2.plot([], [], color="black", label=r"$f^{\star}(X)$")

        ax2.set_xlabel(r"$w_{\perp}^TX$")
        ax2.set_xticks([X2_min, 0, X2_max])
        ax2.set_yticks([0, 0.5, 1])
        ax2.set_yticklabels(["0", "$\\frac{1}{2}$", "1"])
        ax2.axhline(0.5, color="black", lw=0.5)
        ax2.legend()

    fig4 = plt.figure(figsize=(4.5, 6))
    gs = gridspec.GridSpec(2, 1, figure=fig4, height_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])
    Z3 = Z1 + Z2
    crf = ax1.contourf(
        XX0, YY0, Z3, levels=np.linspace(0, 1, 101), cmap=cm, alpha=0.8, vmin=0, vmax=1
    )
    cbar = fig3.colorbar(crf, ax=ax1)
    cbar.ax.set_title(r"$f^{\star}(X)$")
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])

    if w is not None:
        ax1.annotate(
            "",
            xy=(w[0], w[1]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", mutation_scale=20),
        )
        ax1.text(w[0] / 2, w[1] / 2, r"$w$", va="top", ha="left")
        X1 = np.linspace(x_min, x_max, 2)
        X2 = separating_line2D(X1, w, 0)
        ax1.plot(X1, X2, color="black", linestyle=":", label=r"$f(X) = \frac{1}{2}$")

    if w_learned is not None:
        label = "Learned"
        if w_learned[2] != 0:
            X1 = np.linspace(x_min, x_max, 100)
            X2 = separating_line2D(X1, w_learned[1:], w_learned[0])
            (p_line,) = ax1.plot(
                X1, X2, color="tab:orange", linestyle=":", label=label, zorder=11
            )
        else:
            ax1.axvline(0, color="tab:orange", linestyle=":", label=label, zorder=11)
        ax1.annotate(
            "",
            xy=(w_learned[1], w_learned[2]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="tab:orange", mutation_scale=15),
            zorder=11,
        )
        ax1.text(
            w_learned[1] / 2,
            w_learned[2] / 2,
            r"$\beta$",
            va="bottom",
            ha="right",
            color="tab:orange",
        )

    ax1.set_xlim((x_min, x_max))
    ax1.set_ylim((y_min, y_max))
    ax1.set_xticks([x_min, 0, x_max])
    ax1.set_yticks([y_min, 0, y_max])
    ax1.set_title(r"$f^{\star}(X) := f(X) + \Delta(X)$")
    ax1.set_aspect("equal")
    ax1.legend()

    return fig1, fig2, fig3, fig4


def plot_frac_pos_vs_scores(
    frac_pos,
    counts,
    mean_scores,
    y_scores=None,
    y_labels=None,
    bins=None,
    legend_loc="best",
    bbox_to_anchor=None,
    ncol=3,
    xlim_margin=0.15,
    ylim_margin=0.15,
    title=None,
    min_cluster_size=1,
    hist=False,
    k_largest_variance=None,
    k_largest_miscalibration=None,
    ci=None,
    mean_only=False,
    ax=None,
    mean_label=None,
    color_cycler=None,
    xlabel="Confidence score",
    ylabel="Fraction of positives",
    plot_cluster_id=False,
    legend_cluster_sizes=True,
    legend_sizes_only=False,
    vary_cluster_size=True,
    capsize=2,
    cluster_size=None,
    absolute_size_scale=None,
    plot_cal_hist=False,
    figsize=None,
    legend_n_sizes=None,
    legend_min_max=True,
    plot_first_last_bins=True,
    grid_space=0.2,
    legend_title="Cluster sizes",
):
    """Plot fraction of positives in clusters versus the mean scores assigned
    to the clusters, as well as the calibration curve.
    """
    set_latex_font()

    frac_pos = np.array(frac_pos)
    counts = np.array(counts)
    mean_scores = np.array(mean_scores)

    if k_largest_variance is not None and k_largest_miscalibration is not None:
        raise ValueError(
            "Both k_largest_variance and k_largest_miscalibration"
            " should not be passed."
        )

    if k_largest_variance is not None:
        pass  # select the k classes that has the greatest variance

    if k_largest_miscalibration is not None:
        pass  # select the k classes that has the greatest miscalibration

    if frac_pos.ndim >= 3:
        frac_pos = frac_pos.reshape(frac_pos.shape[0], -1)
    if counts.ndim >= 3:
        counts = counts.reshape(counts.shape[0], -1)
    if mean_scores.ndim >= 3:
        mean_scores = mean_scores.reshape(mean_scores.shape[0], -1)

    if frac_pos.shape != counts.shape:
        raise ValueError(
            f"Shape mismatch between frac_pos {frac_pos.shape} and counts {counts.shape}"
        )

    if frac_pos.shape != mean_scores.shape:
        raise ValueError(
            f"Shape mismatch between frac_pos {frac_pos.shape} and mean_scores {mean_scores.shape}"
        )

    available_ci = [None, "clopper", "binomtest"]
    if ci not in available_ci:
        raise ValueError(f"Unkown CI {ci}. Availables: {available_ci}.")

    if hist and ax is not None:
        raise ValueError("Can't specify ax when hist=True.")

    handles = []
    labels = []

    if legend_cluster_sizes and not legend_sizes_only:
        dummy = mpl.patches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False
        )
        handles.append(dummy)
        labels.append("")

    n_bins, n_clusters = frac_pos.shape
    significant_edgecolor = "crimson"
    significant_errcolor = "crimson"
    alpha_nonsignificant = 0.6

    if bins is None:
        bins = np.linspace(0, 1, n_bins + 1)

    prob_bins_na, mean_bins_na = calibration_curve(
        frac_pos, counts, mean_scores, remove_empty=False
    )

    # Remove empty bins
    non_empty = np.sum(counts, axis=1, dtype=float) > 0
    # Remove bins too small:
    big_enough = np.any(counts >= min_cluster_size, axis=1)
    prob_bins = prob_bins_na[non_empty & big_enough]
    mean_bins = mean_bins_na[non_empty & big_enough]

    if min_cluster_size is not None:
        idx_valid_clusters = counts.flatten() >= min_cluster_size
    else:
        idx_valid_clusters = counts.flatten() == counts.flatten()

    cluster_id = np.tile(np.arange(n_clusters), (n_bins, 1))

    df = pd.DataFrame(
        {
            "y_scores": mean_scores.flatten(),
            "y_frac_pos": frac_pos.flatten(),
            "y_size": counts.flatten(),
            "y_prob_bins": np.tile(prob_bins_na, (frac_pos.shape[1], 1)).T.flatten(),
            "y_valid_clusters": idx_valid_clusters,
            "cluster_id": cluster_id.flatten(),
        }
    )

    if hist:
        extra_kwargs = {}
        if figsize is not None:
            a, b = figsize
            if a != b:
                raise ValueError(f"Jointplot will be squared. Given {figsize}.")
            extra_kwargs["height"] = a

        g = sns.JointGrid(
            data=df,
            x="y_scores",
            y="y_frac_pos",
            hue="y_size",
            palette="flare",
            ratio=10,
            space=grid_space,
            **extra_kwargs,
        )
        ax_top = g.figure.axes[1]
        ax = g.figure.axes[0]
        fig = g.fig
        ax_right = g.figure.axes[2]

    elif ax is None:
        fig = plt.figure(figsize=figsize)
        ax_top = plt.gca()
        ax = ax_top

    else:
        fig = ax.figure
        ax_top = ax

    if mean_only:
        if color_cycler is not None:
            ax.set_prop_cycle(color_cycler)
        cal_color = next(ax._get_lines.prop_cycler)["color"]

    else:
        cal_color = "black"

    # Significance
    if ci == "clopper":
        if not mean_only:
            ci_idx = df["y_valid_clusters"]
            ci_count = df["y_frac_pos"][ci_idx] * df["y_size"][ci_idx]
            ci_nobs = df["y_size"][ci_idx]
            ci_low, ci_upp = proportion_confint(
                count=ci_count,
                nobs=ci_nobs,
                alpha=0.05,
                method="beta",
            )
            ci_scores = df["y_scores"][ci_idx]
            ci_frac_pos = df["y_frac_pos"][ci_idx]
            ci_prob_bins = df["y_prob_bins"][ci_idx]

            # Significant cluster are the ones whose CI does not contain
            # the fraction of positive of the bin
            idx_significant = np.logical_or(
                ci_low > ci_prob_bins, ci_upp < ci_prob_bins
            )

            # ci_low and ci_upp are not error widths but CI bounds values
            # the errorbar function requires error widths.
            y_err_low = ci_frac_pos - ci_low
            y_err_upp = ci_upp - ci_frac_pos
            y_err = np.stack([y_err_low, y_err_upp], axis=0)

            p1 = ax.errorbar(
                x=ci_scores[~idx_significant],
                y=ci_frac_pos[~idx_significant],
                yerr=y_err[:, ~idx_significant],
                fmt="none",
                elinewidth=0.5,
                capsize=capsize,
                color="lightgray",
                alpha=alpha_nonsignificant,
                zorder=4,
            )
            p2 = ax.errorbar(
                x=ci_scores[idx_significant],
                y=ci_frac_pos[idx_significant],
                yerr=y_err[:, idx_significant],
                fmt="none",
                elinewidth=0.5,
                capsize=capsize,
                color=significant_errcolor,
                zorder=4,
            )
            if not legend_sizes_only:
                if np.any(idx_significant):
                    handles.append(p2)
                else:
                    handles.append(p1)
                labels.append(r"$95\%$ conf. interval")

        else:
            count_bins = np.sum(counts.reshape(n_bins, -1), axis=1)[non_empty]
            ci_low, ci_upp = proportion_confint(
                count=prob_bins * count_bins,
                nobs=count_bins,
                alpha=0.05,
                method="beta",
            )
            ci_frac_pos = prob_bins
            y_err_low = ci_frac_pos - ci_low
            y_err_upp = ci_upp - ci_frac_pos
            y_err = np.stack([y_err_low, y_err_upp], axis=0)

            ax.plot(mean_bins, ci_upp, color=cal_color, lw=0.5, zorder=1, alpha=0.5)
            ax.plot(mean_bins, ci_low, color=cal_color, lw=0.5, zorder=1, alpha=0.5)
            ax.fill_between(
                mean_bins, ci_low, ci_upp, color=cal_color, zorder=1, alpha=0.05
            )

    elif ci == "binomtest":
        pvalues = []

        ci_idx = df["y_valid_clusters"]
        ci_scores = df["y_scores"][ci_idx]
        ci_frac_pos = df["y_frac_pos"][ci_idx]
        ci_prob_bins = df["y_prob_bins"][ci_idx]
        ci_count = df["y_frac_pos"][ci_idx] * df["y_size"][ci_idx]
        ci_nobs = df["y_size"][ci_idx]

        for i in range(len(ci_scores)):
            res = scipy.stats.binomtest(
                k=int(ci_count.iloc[i]),
                n=int(ci_nobs.iloc[i]),
                p=ci_prob_bins.iloc[i],
            )

            pvalues.append(res.pvalue)

        pvalues = pd.Series(pvalues, index=ci_scores.index)

        idx_significant = pvalues < 0.05

    else:
        idx_significant = None

    def scatter_with_size(x, y, hue, valid_clusters, significant_clusters, cluster_id):
        x = x[valid_clusters]
        y = y[valid_clusters]
        hue = hue[valid_clusters]

        min_scatter_size = 25
        max_scatter_size = 80

        cmap = sns.color_palette("flare", as_cmap=True)

        min_cluster_size = np.min(hue)
        max_cluster_size = np.max(hue)

        if absolute_size_scale is not None:
            vmin, vmax = absolute_size_scale
            if vmin is None:
                vmin = min_cluster_size
            if vmax is None:
                vmax = max_cluster_size
        else:
            vmin = min_cluster_size
            vmax = max_cluster_size

        if vary_cluster_size:
            size = hue
            sizes = (min_scatter_size, max_scatter_size)

            size_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            size = None
            sizes = None
            size_norm = None

        hue_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        extra_kwargs = {}
        if cluster_size is not None:
            extra_kwargs["s"] = cluster_size

        g = sns.scatterplot(
            x=x,
            y=y,
            size=size,
            hue=hue,
            hue_norm=hue_norm,
            sizes=sizes,
            size_norm=size_norm,
            palette=cmap,
            zorder=5,
            alpha=alpha_nonsignificant if significant_clusters is not None else 1,
            legend="auto" if legend_cluster_sizes else False,
            **extra_kwargs,
        )
        if significant_clusters is not None:
            g = sns.scatterplot(
                x=x[significant_clusters],
                y=y[significant_clusters],
                size=size[significant_clusters] if size is not None else None,
                hue=hue[significant_clusters],
                hue_norm=hue_norm,
                sizes=sizes,
                size_norm=size_norm,
                palette=sns.color_palette("flare", as_cmap=True),
                zorder=5,
                edgecolor=significant_edgecolor,
                linewidth=0.75,
                legend=False,
                **extra_kwargs,
            )

        if legend_cluster_sizes:
            # Add minimum and maximum cluster sizes in legend handles and labels
            H, L = g.get_legend_handles_labels()
            cmap = sns.color_palette("flare", as_cmap=True)

            _handles = handles[:]
            _labels = labels[:]
            handles.clear()
            labels.clear()

            min_str = str(int(np.min(hue)))
            max_str = str(int(np.max(hue)))

            if min_str != L[0]:
                handles.append(copy(H[0]))
                if vary_cluster_size:
                    handles[0].set_sizes([min_scatter_size])
                handles[0].set_facecolors([cmap(hue_norm(min_cluster_size))])
                handles[0].set_edgecolors([cmap(hue_norm(min_cluster_size))])

                s = " (min)" if legend_min_max else ""
                labels.append(f"{min_str}{s}")

            if legend_n_sizes is None or legend_n_sizes + 2 >= len(H):
                choices = np.arange(len(H))
            else:
                choices = np.linspace(0, len(H), legend_n_sizes + 2, dtype=int)
                choices = choices[1:-1]

            for i in choices:
                handles.append(H[i])
                labels.append(L[i])

            if max_str != L[-1]:
                handles.append(copy(H[-1]))
                if vary_cluster_size:
                    handles[-1].set_sizes([max_scatter_size])
                handles[-1].set_facecolors([cmap(hue_norm(max_cluster_size))])
                handles[-1].set_edgecolors([cmap(hue_norm(max_cluster_size))])

                s = " (max)" if legend_min_max else ""
                labels.append(f"{max_str}{s}")

            handles.extend(_handles)
            labels.extend(_labels)

        if plot_cluster_id:
            cluster_id = cluster_id[valid_clusters]
            for i in range(len(x)):
                ax.annotate(
                    cluster_id.iloc[i],
                    (x.iloc[i], y.iloc[i]),
                    zorder=6,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize="xx-small",
                )

        return g

    def histplot_with_size(x, vertical, hue):
        color_hue = np.ones_like(x)
        if vertical:
            x, y = None, x
        else:
            x, y = x, None
        sns.histplot(
            x=x,
            y=y,
            weights=hue,
            hue=color_hue,
            palette="flare",
            bins=list(bins),
            legend=False,
        )

    if hist:
        if not mean_only:
            g.plot_joint(
                scatter_with_size,
                valid_clusters=df["y_valid_clusters"],
                significant_clusters=idx_significant,
                cluster_id=df["cluster_id"],
            )
        g.plot_marginals(histplot_with_size)

    elif not mean_only:
        g = scatter_with_size(
            df["y_scores"],
            df["y_frac_pos"],
            df["y_size"],
            valid_clusters=df["y_valid_clusters"],
            significant_clusters=idx_significant,
            cluster_id=df["cluster_id"],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    bins = np.linspace(0, 1, n_bins + 1)
    if not plot_first_last_bins:
        bins = bins[1:-1]

    for x in bins:
        ax.axvline(x, lw=0.5, ls="--", color="grey", zorder=-1)

    # Plot calibration curve
    (p0,) = ax.plot([0, 1], [0, 1], ls="--", lw=1, color="black", zorder=0)
    if mean_only:
        marker = "o"
        markeredgecolor = "white"
        markeredgewidth = 0.1
    else:
        marker = "."
        markeredgecolor = None
        markeredgewidth = None

    (p1,) = ax.plot(
        mean_bins,
        prob_bins,
        marker=marker,
        markersize=5,
        color=cal_color,
        label=mean_label,
        zorder=2,
        markeredgecolor=markeredgecolor,
        markeredgewidth=markeredgewidth,
    )

    if not legend_sizes_only:
        handles.append(p0)
        labels.append("Perfect calibration")
        handles.append(p1)
        labels.append("Calibration curve")

    # Plot background histogram for calibration
    if plot_cal_hist:
        x = ((0.5 + np.arange(n_bins)) / n_bins)[non_empty]
        y = prob_bins
        ax.bar(
            x,
            height=y,
            width=1 / n_bins,
            color=(0.85, 0.85, 0.85, 1),
            zorder=0,
            edgecolor=(0.5, 0.5, 0.5, 1),
        )

    if not mean_only and len(handles) > 0:
        legend = ax.legend(
            loc=legend_loc,
            ncol=ncol,
            bbox_to_anchor=bbox_to_anchor,
            handles=handles,
            labels=labels,
            fancybox=True,
            framealpha=1,
        )
        if legend_cluster_sizes:
            legend.set_title(legend_title)
    if xlim_margin is not None:
        ax.set_xlim((-xlim_margin, 1 + xlim_margin))
    if ylim_margin is not None:
        ax.set_ylim((-ylim_margin, 1 + ylim_margin))

    ax.set_aspect("equal")
    if title is not None:
        ax_top.set_title(title)

    return fig


def barplot_ece_gl(
    net,
    val,
    acc,
    ax=None,
    legend=True,
    bbox_to_anchor=(1, 1),
    loc="upper left",
    append_acc=True,
    table_acc=False,
    table_fontsize=12,
    ax_ratio=1,
    gray_bg=None,
    ncol=1,
    detailed=False,
    plot_xlabel=False,
):
    cl, glexp, glexp_bias, glind, clind = val
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    acc = np.array(acc)
    idx_sort = np.argsort(-acc)
    if append_acc:
        net = np.array([f"{n} (Acc={100*a:.1f}%)" for n, a in zip(net, acc)])
    else:
        net = np.array(net)

    label_glexp = r"$\widehat{\mathrm{GL}}_{explained}(S_B)$"
    label_cl = r"$\widehat{\mathrm{CL}}(S_B)$"
    label_bias = "bias"
    label_glind = r"$\widehat{\mathrm{GL}}_{induced}$"
    order = net[idx_sort]

    if detailed:
        sns.barplot(
            x=glexp + cl,
            y=net,
            ax=ax,
            edgecolor="tab:blue",
            order=order,
            label=label_cl,
            facecolor="tab:blue",
            lw=2,
        )
        sns.barplot(
            x=glexp,
            y=net,
            ax=ax,
            edgecolor="tab:red",
            order=order,
            label=label_glexp,
            facecolor="tab:red",
            lw=2,
        )
        sns.barplot(
            x=glexp_bias + glind,
            y=net,
            ax=ax,
            edgecolor="none",
            order=order,
            label=label_bias,
            facecolor="tab:orange",
            lw=2,
        )
        sns.barplot(
            x=glind,
            y=net,
            ax=ax,
            edgecolor="none",
            order=order,
            label=label_glind,
            facecolor="tab:green",
            lw=2,
        )
        xlabel = f"{label_glexp} + {label_cl}"

    else:
        glexp_corrected = glexp - glexp_bias - glind
        cl_corrected = cl - clind
        label_cl = r"$\widehat{\mathrm{CL}}$"
        label_glexp_corr = r"$\widehat{\mathrm{GL}}_{\mathrm{LB}}$"
        sns.barplot(
            x=glexp_corrected + cl_corrected,
            y=net,
            ax=ax,
            edgecolor="tab:blue",
            order=order,
            label=label_cl,
            facecolor="tab:blue",
            lw=2,
        )
        sns.barplot(
            x=glexp_corrected,
            y=net,
            ax=ax,
            edgecolor="tab:red",
            order=order,
            label=label_glexp_corr,
            facecolor="tab:red",
            lw=2,
        )
        xlabel = f"{label_glexp_corr} + {label_cl}"

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        ax.legend(
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            ncol=ncol,
            handles=handles,
            labels=labels,
        )

    if not plot_xlabel:
        xlabel = None
    ax.set_xlabel(xlabel)
    ax.set_ylabel(None)

    cellText = np.transpose([list([f"{100*a:.1f}" for a in acc[idx_sort]])])
    n_nets = len(net)

    if gray_bg is not None:
        ylim = ax.get_ylim()
        cellColours = [["white"]] * n_nets
        for i in range(0, n_nets, 2):
            cellColours[i] = [gray_bg]
        for k in range(0, n_nets, 2):
            ax.axhspan(k - 0.5, k + 0.5, color=gray_bg, zorder=0)
        ax.set_ylim(ylim)
    else:
        cellColours = None

    if table_acc:
        table_width = 0.14 * ax_ratio
        table_xpos = 1.02 + 0.025 * ax_ratio
        xpos = table_xpos + table_width + 0.02
        ypos = 0.5
        ax.annotate(
            r"Accuracy$\uparrow$ (\%)",
            xy=(xpos, ypos),
            xytext=(xpos, ypos),
            xycoords="axes fraction",
            ha="left",
            va="center",
            fontsize=plt.rcParams["axes.labelsize"],
            rotation=-90,
        )
        table = ax.table(
            cellText=cellText,
            loc="right",
            rowLabels=None,
            colLabels=None,
            bbox=[table_xpos, 0, table_width, 1],
            cellColours=cellColours,
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(table_fontsize)

    return ax


def barplots_ece_gl_cal(
    net,
    val1,
    val2,
    acc,
    plot_table=True,
    keep_scale=True,
    figsize=(4, 3.5),
    loc="center right",
    bbox_to_anchor=(1, 0.5),
):
    set_latex_font()

    # Font sizes
    gray_bg = ".96"
    table_fontsize = 11
    plt.rc("ytick", labelsize=14)
    plt.rc("axes", labelsize=16)
    plt.rc("axes", titlesize=18)
    plt.rc("legend", fontsize=14)
    plt.rc("legend", borderpad=0.3)
    plt.rc("legend", borderaxespad=0.1)
    plt.rc("legend", handlelength=1.6)
    plt.rc("legend", labelspacing=0.4)
    plt.rc("legend", handletextpad=0.4)

    ax_ratio = 4
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [ax_ratio, 1]}
    )
    plt.subplots_adjust(wspace=0.05)
    ax1 = axes[0]
    ax2 = axes[1]

    barplot_ece_gl(
        net,
        val1,
        acc,
        ax1,
        bbox_to_anchor=bbox_to_anchor,
        loc=loc,
        append_acc=False,
        ncol=1,
        gray_bg=gray_bg,
        plot_xlabel=True,
    )

    if val2 is not None:
        barplot_ece_gl(
            net,
            val2,
            acc,
            ax2,
            legend=False,
            table_acc=plot_table,
            ax_ratio=ax_ratio,
            table_fontsize=table_fontsize,
            gray_bg=gray_bg,
            plot_xlabel=False,
        )

    ax1.set_xlim(0, 0.22)
    xmin, xmax = ax1.get_xlim()
    if keep_scale:
        xtick = xmin + 0.23 * (xmax - xmin)
        xtick = float(f"{xtick:.1g}")
        ax2.set_xticks([0, xtick])
        ax2.set_xlim((xmin, xmax / ax_ratio))
    ax2.get_yaxis().set_visible(False)
    ax1.set_title("No recalibration")
    ax2.set_title("Isotonic")

    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    return fig


def plot_fig_theorem_v2(isoline_right=True, squared=True, legend_right=True):
    set_latex_font(extra_preamble=[r"\usepackage[mathscr]{eucal}"])

    plt.rc("legend", borderpad=0.1)
    plt.rc("legend", borderaxespad=0.1)
    plt.rc("legend", labelspacing=0.3)

    fig = plt.figure(figsize=(3.7, 3.7))
    ax = plt.gca()
    colors = sns.color_palette("hls", 8).as_hex()
    fontsize = 14
    color_cluster1 = colors[5]
    color_cluster2 = colors[0]
    xmin, xmax = 0.14, 0.93
    if squared:
        ymin, ymax = 0.15, 0.94  # For the squared version
    else:
        isoline_right = legend_right
        ymin, ymax = 0.15, 0.735
    x_e = 0.48
    x_er1 = 0.43
    x_er2 = 0.8
    x_mid = 0.54
    x_r1 = 0.52
    x_r2 = 0.49
    p_under = 0.6
    p_above = 0.8
    p_mid = 0.5 * (p_above + p_under)

    a = -1.6666666666666521e-001
    b = 5.6666666666666499e-001
    c = 0.1

    width = 0.2
    X = np.linspace(0, 1, 1000)

    def curve1(X):
        return a * X**2 + b * X + c

    def curve2(X):
        return curve1(X) + width

    def cut(X):
        return 1.6 - 2.3 * X

    def is_in_cluster1(X, margin=0):
        x, y = X
        return (
            y >= curve1(x) + margin and y <= curve2(x) - margin and y <= cut(x) - margin
        )

    def is_in_cluster2(X, margin=0):
        x, y = X
        return (
            y >= curve1(x) + margin and y <= curve2(x) - margin and y > cut(x) + margin
        )

    def is_in_cluster1_or_2(X, margin=0):
        x, y = X
        return is_in_cluster1(X, margin) or is_in_cluster2(X, margin)

    def is_out_frame(X, margin=0):
        x, y = X
        return x <= margin or x >= 1 - margin or y <= margin or y >= 1 - margin

    Y1 = curve1(X)
    Y2 = curve2(X)
    Y2 = curve2(X)
    Y_cut = cut(X)

    (line,) = ax.plot(X, Y1, color="black", label="Level set $S = 0.7$")
    ax.plot(X, Y2, color="black")

    h = 100
    qmin = 0.5
    qmax = 0.9
    qmid = (qmin + qmax) / 2
    n_levels = 500

    XX = np.linspace(xmin, xmax, h)
    WW = np.linspace(0, 0.2, h)

    XX0, WW0 = np.meshgrid(XX, WW)
    YY0 = curve1(XX0) + WW0

    def Q(X):
        x, y = X[:, 0], X[:, 1]
        v_perp = np.stack([np.ones_like(x), 2 * a * x + b], axis=1)
        t = np.sum(X * v_perp, axis=1)
        tmax = np.max(t)
        tmin = np.min(t)
        q = (qmax - qmin) / (tmax - tmin) * (t - tmin) + qmin
        return q

    Z2 = Q(np.c_[XX0.ravel(), YY0.ravel()])
    Z2 = Z2.reshape(XX0.shape)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cm = plt.cm.RdBu_r
    crf = ax.contourf(
        XX0,
        YY0,
        Z2,
        levels=np.linspace(qmin, qmax, n_levels),
        cmap=cm,
        alpha=1,
        vmin=qmin,
        vmax=qmax,
    )
    cbar = fig.colorbar(crf, cax=cax)
    cbar.set_ticks([qmin, qmax])
    cbar.ax.set_yticklabels([f"{qmin:.2g}", f"{qmax:.2g}"])

    for collection in crf.collections:
        collection.set_edgecolor("face")
        collection.set_linewidth(0.02)

    idx_cut_visible = (Y1 <= Y_cut) & (Y_cut <= Y2)
    ax.plot(X[idx_cut_visible], Y_cut[idx_cut_visible], color="black", lw=1, ls=":")

    x_isoline = xmax if isoline_right else xmin
    delta = 0.01 if isoline_right else -0.01
    deltay = 0 if squared else 0.01
    deltay = deltay if legend_right else -0.005
    ha = "left" if isoline_right else "right"
    xy_low = (x_isoline, curve1(x_isoline) + deltay)
    xy_up = (x_isoline, curve2(x_isoline) + deltay)
    xy_mid = (x_isoline, 0.5 * (curve1(x_isoline) + curve2(x_isoline) + deltay))
    xytext_low = (x_isoline + delta, curve1(x_isoline) + deltay)
    xytext_up = (x_isoline + delta, curve2(x_isoline) + deltay)
    xytext_mid = (
        x_isoline + delta,
        0.5 * (curve1(x_isoline) + curve2(x_isoline) + deltay),
    )
    va = "center"
    dd = 0.005
    ax.annotate(
        rf"$\mathbb{{E}}[Q|S] = {p_mid}$",
        xy=(x_e, curve2(x_e)),
        xytext=(x_e - dd, curve2(x_e) + dd),
        va="bottom",
        ha="right",
        color="black",
        fontsize=fontsize,
    )
    ax.annotate(
        rf"$\mathbb{{E}}[Q|S,\mathscr{{R}}_1] = {p_under}$",
        xy=(x_er1, curve1(x_er1)),
        xytext=(x_er1 + 2 * dd, curve1(x_er1) - 2 * dd),
        va="top",
        ha="left",
        color=color_cluster1,
        fontsize=fontsize,
    )
    ax.annotate(
        rf"$\mathbb{{E}}[Q|S,\mathscr{{R}}_2] = {p_above}$",
        xy=(x_er2, curve2(x_er2)),
        xytext=(x_er2 - dd, curve2(x_er2)),
        va="bottom",
        ha="right",
        color=color_cluster2,
        fontsize=fontsize,
    )
    ax.annotate(
        r"$\mathscr{{R}}_1$",
        xy=(x_r1, curve2(x_r1)),
        xytext=(x_r1 - dd, curve1(x_r1) + dd),
        va="bottom",
        ha="right",
        color="black",
        fontsize=fontsize,
    )
    ax.annotate(
        r"$\mathscr{{R}}_2$",
        xy=(x_r2, curve2(x_r2)),
        xytext=(x_r2 + dd, curve2(x_r2) - dd),
        va="top",
        ha="left",
        color="black",
        fontsize=fontsize,
    )

    n = 12
    m = 10
    rs = np.random.RandomState(0)
    y_under = rs.binomial(n=1, p=p_under, size=n)
    y_above = rs.binomial(n=1, p=p_above, size=n)

    mean1 = np.array([0.48, 0.32])
    mean2 = np.array([0.45, 0.55])
    cov1 = 1 / 180 * np.diag([5, 1])
    cov2 = 1 / 140 * np.diag([7, 1])
    p1 = 0
    p2 = 1

    # Rotate
    def rotation(cov, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return R @ cov @ R.T

    theta1 = np.pi / 12
    theta2 = np.pi / 12
    cov1 = rotation(cov1, theta1)
    cov2 = rotation(cov2, theta2)

    def sample_x():
        if rs.binomial(n=1, p=0.5):
            return rs.multivariate_normal(mean1, cov1), rs.binomial(n=1, p=p1)
        return rs.multivariate_normal(mean2, cov2), rs.binomial(n=1, p=p2)

    L_X_dist1 = []  # samples in dist 1
    L_X_dist2 = []  # samples in dist 2
    L_X_cluster1 = []
    L_X_cluster2 = []
    L_y = []
    L_y_cluster1 = []
    L_y_cluster2 = []
    while (
        len(L_X_dist1) < n
        or len(L_X_dist2) < n
        or len(L_X_cluster1) < m
        or len(L_X_cluster2) < m
    ):
        x_prop, y_prop = sample_x()

        if is_out_frame(x_prop, margin=0.02):
            continue

        if is_in_cluster1(x_prop):
            if is_in_cluster1(x_prop, margin=0.02) and len(L_X_cluster1) < m:
                if np.sum(L_y_cluster1) >= int(p_under * m) and y_prop == 1:
                    continue  # ignore sample with label 1 because too many already
                if (
                    len(L_y_cluster1) - np.sum(L_y_cluster1) >= m - int(p_under * m)
                    and y_prop == 0
                ):
                    continue  # ignore sample with label 0 because too many already
                L_X_cluster1.append(x_prop)
                L_y_cluster1.append(y_prop)
            continue

        if is_in_cluster2(x_prop):
            if is_in_cluster2(x_prop, margin=0.02) and len(L_X_cluster2) < m:
                if np.sum(L_y_cluster2) >= int(p_above * m) and y_prop == 1:
                    continue  # ignore sample with label 1 because too many already
                if (
                    len(L_y_cluster2) - np.sum(L_y_cluster2) >= m - int(p_above * m)
                    and y_prop == 0
                ):
                    continue  # ignore sample with label 0 because too many already
                L_X_cluster2.append(x_prop)
                L_y_cluster2.append(y_prop)
            continue

        # Ignore points too close to clusters bondaries
        if is_in_cluster1_or_2(x_prop, margin=-0.02):
            continue

        if len(L_X_dist1) < n:
            L_X_dist1.append(x_prop)
            L_y.append(y_prop)

        elif len(L_X_dist2) < n:
            L_X_dist2.append(x_prop)
            L_y.append(y_prop)

    assert len(L_X_dist2) == n
    assert len(L_X_dist1) == n
    assert len(L_y) == 2 * n

    Xs = np.array(L_X_dist1 + L_X_dist2)
    y_labels = np.array(L_y)
    Xs_clusters = np.array(L_X_cluster1 + L_X_cluster2)
    y_labels_clusters1 = np.array(L_y_cluster1)
    y_labels_clusters2 = np.array(L_y_cluster2)
    y_labels_clusters = np.concatenate([y_labels_clusters1, y_labels_clusters2])

    assert Xs.shape == (2 * n, 2)
    assert y_labels.shape == (2 * n,)
    assert y_labels_clusters.shape == (2 * m,)

    idx_pos = y_labels == 1
    idx_pos = y_labels_clusters == 1

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if squared:
        legend = ax.legend(loc="upper center", ncol=2)
    else:
        bbox_to_anchor = (1, -0.02) if legend_right else (0.0, 1.02)
        loc = "lower left" if legend_right else "upper right"
        handles = [
            Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=line.get_linewidth(),
                label=line.get_label(),
            )
        ]
        legend = ax.legend(
            loc="lower right",
            ncol=1,
            bbox_to_anchor=(1, 0),
            fancybox=False,
            framealpha=0,
            handles=handles,
        )
    frame = legend.get_frame()
    frame.set_linewidth(0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")

    # Plot Xcal feature space in corner
    ax.annotate(
        r"$\mathcal{X}$",
        xy=(0.02, 0.972),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=fontsize + 3,
    )
    cbar.ax.annotate(
        r"$Q$",
        xy=(1.6, 0.5),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=fontsize,
    )

    ax.add_patch(
        plt.Rectangle(
            (0, 0.86),
            0.115,
            0.14,
            linewidth=0.7,
            fill=False,
            color="black",
            alpha=1,
            zorder=1000,
            transform=ax.transAxes,
            figure=fig,
        )
    )

    return fig


def plot_fig_counter_example(with_arrow=False):
    set_latex_font()

    fontsize = 11
    fontsize_threshold = fontsize - 2.5
    lw = 2
    plt.rc("legend", fontsize=10)
    plt.rc("legend", handletextpad=0.5)
    plt.rc("legend", columnspacing=1.3)
    plt.rc("legend", borderpad=0.3)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", handlelength=1.5)
    plt.rc("legend", labelspacing=0.3)
    plt.rc("axes", labelsize=11)
    figsize = (1.8, 1.8)

    x_min = -1
    x_max = 1
    n = 100

    def f(x):
        x = np.atleast_1d(x)
        y = np.zeros_like(x)
        y[x > 0] = 0.7
        y[x <= 0] = 0.2
        return y

    def f_star(x):
        x = np.atleast_1d(x)
        y = np.zeros_like(x)

        y[x > 0] = 0.6
        y[x > 0.5] = 0.8
        y[x <= 0] = 0.3
        y[x < -0.5] = 0.1
        return y

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    disc_gap = 0.01

    ax.axhline(0.5, color="black", lw=0.5)
    X = np.linspace(x_min, x_max, n)
    S = f(X)
    X, S = insert_nan_at_discontinuities(X, S, min_gap=disc_gap)
    ax.plot(X, S, label="$S(X)$", color="black", lw=lw)
    Q = f_star(X)
    X, Q = insert_nan_at_discontinuities(X, Q, min_gap=disc_gap)
    ax.plot(X, Q, label="$Q(X)$", color="tab:red", ls="-", lw=lw)
    ax.set_xlabel(r"$X \sim U([-1, 1])$")
    ax.set_ylabel("Output")
    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.set_xticks([-1, 1])
    ax.set_yticks([0, 0.5, 1])
    h2, l2 = ax.get_legend_handles_labels()

    delta_left = 0.05
    delta_right = 0.05

    ax.annotate(
        "0.6",
        xy=(1, 0.6),
        xytext=(1 + delta_right, 0.6),
        color="tab:red",
        va="center",
        ha="left",
        fontsize=fontsize,
    )
    ax.annotate(
        "0.7",
        xy=(1, 0.7),
        xytext=(1 + delta_right, 0.7),
        color="black",
        va="center",
        ha="left",
        fontsize=fontsize,
    )
    ax.annotate(
        "0.8",
        xy=(1, 0.8),
        xytext=(1 + delta_right, 0.8),
        color="tab:red",
        va="center",
        ha="left",
        fontsize=fontsize,
    )

    arrowprops = dict(arrowstyle=f"->, head_length=0, head_width=0", lw=1)
    ax.annotate(
        "0.1",
        xy=(-1, 0.1),
        xytext=(-1 - delta_left, 0.1),
        color="tab:red",
        va="center",
        ha="right",
        fontsize=fontsize,
        arrowprops=arrowprops,
    )
    ax.annotate(
        "0.2",
        xy=(-1, 0.2),
        xytext=(-1 - delta_left, 0.2),
        color="black",
        va="center",
        ha="right",
        fontsize=fontsize,
        arrowprops=arrowprops,
    )
    ax.annotate(
        "0.3",
        xy=(-1, 0.3),
        xytext=(-1 - delta_left, 0.3),
        color="tab:red",
        va="center",
        ha="right",
        fontsize=fontsize,
        arrowprops=arrowprops,
    )

    if with_arrow:
        x_text = -0.55
        y_text = 0.43
        ax.annotate(
            "decision threshold",
            xy=(x_text, y_text),
            xytext=(x_text, y_text),
            color="black",
            va="center",
            ha="left",
            fontsize=fontsize_threshold,
        )
        style = "Simple, tail_width=0.01, head_width=2, head_length=3"
        kw = dict(arrowstyle=style, color="k")
        a = patches.FancyArrowPatch(
            (x_text, y_text), (-0.98, 0.495), connectionstyle="arc3,rad=-.22", **kw
        )
        ax.add_patch(a)

    else:
        ax.annotate(
            "decision threshold",
            xy=(0, 0.5),
            xytext=(0, 0.49),
            color="black",
            va="top",
            ha="center",
            fontsize=fontsize_threshold,
        )

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.81), ncol=2)
    ax.set_xlim(-1, 1)

    return fig


def plot_renditions_calibration(df, x="diff", y="rendition", hue="net"):
    set_latex_font()
    dfgb = df[[x, y]].groupby([y]).mean().reset_index()
    order = dfgb.sort_values(x, ascending=False)[y]

    plt.rc("legend", fontsize=11)
    plt.rc("legend", title_fontsize=12)
    plt.rc("legend", handletextpad=0.5)
    plt.rc("legend", columnspacing=1.3)
    plt.rc("legend", borderpad=0.3)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", handlelength=1.5)
    plt.rc("legend", labelspacing=0.1)
    plt.rc("xtick", labelsize=7)
    plt.rc("ytick", labelsize=11)
    plt.rc("axes", labelsize=12)

    n_renditions = len(np.unique(df[y]))
    np.random.seed(0)
    g = sns.catplot(data=df, x=x, y=y, hue=hue, order=order, height=3.5)
    fig = g.figure
    ax = fig.axes[0]
    sns.stripplot(
        data=dfgb, x=x, y=y, color="black", ax=ax, jitter=0, order=order, legend=False
    )

    ax.axvline(0, color="darkgray", lw=1, zorder=0)
    xmin, xmax = ax.get_xlim()
    xabs = max(abs(xmin), abs(xmax))
    ax.set_ylim((n_renditions - 0.5, 0.5))

    # Add gray layouts in the background every other rows
    for k in range(1, n_renditions, 2):
        ax.axhspan(k - 0.5, k + 0.5, color=".93", zorder=-1)

    ax.set_xlabel(r"$\bar{C}_{rendition} - \bar{C}_{all}$")
    ax.set_ylabel("Renditions")
    g.legend.set_title("Network")

    return fig


#  Grouping loss
def plot_simu(
    df, x="n_samples_per_cluster_per_bin", legend=True, only_strat="uniform", ax=None
):
    set_latex_font()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    else:
        fig = ax.figure
    df["LB_est"] = df["LB_biased"] - df["bias"] - df["GL_ind"]

    idx_means = {}
    for strategy, ls in [("uniform", "-"), ("quantile", "--")]:
        subdf = df.query("n_size_one_clusters > 0 and strategy == @strategy")
        idx = subdf.groupby("trial").aggregate({x: max})
        idx_mean = idx.mean().item()
        idx_means[strategy] = idx_mean

    idx_mean_max = np.nanmax(list(idx_means.values()))

    _df = df.melt(
        id_vars=[x, "strategy", "trial", "n_size_one_clusters"],
        value_vars=["LB_biased", "bias", "GL_ind", "LB_est", "GL"],
    )

    # Discard points of GL_LB that are mainly negative
    for strategy in ["uniform", "quantile"]:
        __df = _df.query('variable == "LB_est" and value < 0 and strategy == @strategy')
        x_to_discard, counts = np.unique(__df[x], return_counts=True)
        x_to_discard = x_to_discard[counts >= 0.5 * __df.shape[0]]
        _df = _df.query(
            f'{x} not in @x_to_discard or strategy != @strategy or variable != "LB_est"'
        )

    # Filter to eliminate parts of curves that are invalid
    idx_uniform = int(np.nan_to_num(idx_means["uniform"]))
    idx_quantile = int(np.nan_to_num(idx_means["quantile"]))
    _df = _df.query(
        f'strategy == "uniform" and {x} >= {idx_uniform}'
        f" or "
        f'strategy == "quantile" and {x} >= {idx_quantile}'
        f" or "
        f'variable in ["LB_biased", "GL_ind", "GL"]'
    )

    if only_strat is not None:
        _df = _df.query(f'strategy == "{only_strat}"')

    _df["strategy"] = _df["strategy"].replace(
        {
            "uniform": "Equal-width",
            "quantile": "Equal-mass",
        }
    )
    var_replace = {
        "LB_biased": r"$\widehat{\mathrm{GL}}_{\mathit{plugin}}$",
        "bias": r"$\widehat{\mathrm{GL}}_{\mathit{bias}}$",
        "GL_ind": r"$\widehat{\mathrm{GL}}_{\mathit{induced}}$",
        "LB_est": r"$\widehat{\mathrm{GL}}_{\mathrm{LB}}$",
        "GL": r"True $\mathrm{GL}$",
    }
    _df["variable"] = _df["variable"].replace(var_replace)
    _df.rename({"strategy": "Binning"}, axis=1, inplace=True)

    style = None if only_strat is not None else "Binning"
    style_order = (None if only_strat is not None else ["Equal-width", "Equal-mass"],)

    hue_order = [
        var_replace[v]
        for v in [
            "GL",
            "LB_biased",
            "bias",
            "GL_ind",
            "LB_est",
        ]
    ]

    palette = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]

    sns.lineplot(
        data=_df,
        x=x,
        y="value",
        hue="variable",
        style=style,
        style_order=style_order,
        ax=ax,
        errorbar=("sd", 1),
        legend="auto" if legend else False,
        palette=palette,
        hue_order=hue_order,
        err_kws=dict(edgecolor="none"),
    )

    if not np.isnan(idx_mean_max):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_betweenx(
            [-1, 1],
            -idx_mean_max,
            idx_mean_max,
            color="lightgray",
            edgecolor="none",
            alpha=0.5,
            zorder=0,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        plt.rc("legend", borderaxespad=0.1)

        if style_order is None:
            handles = handles[1:]
            labels = labels[1:]

        _legend = ax.legend(
            handles=handles,
            labels=labels,
            ncol=1,
            bbox_to_anchor=(1, 1),
            loc="upper left",
            fancybox=False,
            framealpha=1,
        )
        frame = _legend.get_frame()
        frame.set_linewidth(0)
    ax.set(ylabel=None)

    return fig


def plot_fig_binning(N=1000, n_bins=2):
    set_latex_font()
    plt.rc("legend", borderpad=0.4)
    plt.rc("legend", borderaxespad=0.1)
    plt.rc("legend", columnspacing=1.2)
    plt.rc("legend", handletextpad=0.5)

    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.8))
    plot_first_last_bins = False

    # Plot bins
    bins = np.linspace(0, 1, n_bins + 1)
    _bins = bins if plot_first_last_bins else bins[1:-1]
    for i, x in enumerate(_bins):
        label = "Bin edge" if i == 0 else None
        ax.axvline(x, lw=0.5, ls="--", color="grey", zorder=-1, label=label)

    # Plot calibration curve
    S = np.linspace(0, 1, N)

    def c(s):
        return np.square(s)

    C = c(S)
    ax.plot(S, C, color="black", label="$C$")

    # Plot binned calibration curve
    for i in range(n_bins):
        a = bins[i]
        b = bins[i + 1]
        CB = 1 / (3 * (b - a)) * (b**3 - a**3)
        label = r"$C_B$" if i == 0 else None
        ax.plot([a, b], [CB, CB], color="black", label=label, ls="--")

        label = "$S_B$" if i == 0 else None
        line = ax.scatter((a + b) / 2, 0, color="black", label=label)
        line.set_clip_on(False)

        Sab = np.linspace(a, b, N // n_bins)
        M = len(Sab)

        label = r"$\mathrm{GL}_{induced}$" if i == 0 else None
        ax.fill_between(
            Sab,
            c(Sab),
            [CB] * M,
            color="tab:red",
            label=label,
            edgecolor="none",
            zorder=-2,
        )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    d = 0.08
    ax.annotate(
        "$S$",
        xy=(0.5, -d),
        xytext=(0.5, -d),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=plt.rcParams["legend.fontsize"],
    )

    handles, labels = ax.get_legend_handles_labels()
    handles = list(np.roll(handles, 1))
    labels = list(np.roll(labels, 1))
    handles[0], handles[1] = handles[1], handles[0]
    labels[0], labels[1] = labels[1], labels[0]
    _legend = ax.legend(
        ncol=1,
        framealpha=0,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        handles=handles,
        labels=labels,
    )

    frame = _legend.get_frame()
    frame.set_linewidth(0)
    return fig
