"""Code that generates the figures of the paper."""
import itertools
import os
import re
from itertools import product
from os.path import join
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from joblib import Memory, Parallel, delayed
import torch
from matplotlib.ticker import MultipleLocator
from sklearn.utils import check_random_state
from tqdm import tqdm

from src._linalg import create_orthonormal_vector
from src._plot import (
    barplots_ece_gl_cal,
    plot_ffstar_1d,
    plot_ffstar_2d_v2,
    plot_fig_binning,
    plot_fig_counter_example,
    plot_fig_theorem_v2,
    plot_frac_pos_vs_scores,
    plot_renditions_calibration,
    plot_score_vs_probas2,
    plot_simu,
)
from src._utils import (
    _get_out_kwargs,
    bin_train_test_split,
    binarize_multiclass_max,
    calibrate_scores,
    compute_classif_metrics,
    compute_multi_classif_metrics,
    save_fig,
    save_path,
)
from src.grouping_loss import (
    CEstimator,
    calibration_curve,
    compute_calib_metrics,
    estimate_GL_induced,
    grouping_loss_lower_bound,
)
from src.networks import (
    ALL_IMAGENET_NETWORKS,
    IMAGENET_VGG,
    BaseNet,
    IMAGENET_AlexNet,
    IMAGENET_ConvNeXt,
    IMAGENET_DenseNet,
    IMAGENET_EfficientNet,
    IMAGENET_GoogLeNet,
    IMAGENET_Inception,
    IMAGENET_MNASNet,
    IMAGENET_MobileNet,
    IMAGENET_RegNet,
    IMAGENET_ResNet,
    IMAGENET_ResNext,
    IMAGENET_ShuffleNet,
    IMAGENET_VisionTransformer,
    IMAGENET_WideResNet,
    ZeroShotBartYahoo,
)
from src.partitioning import (
    cluster_evaluate,
    cluster_evaluate_marginals,
    cluster_evaluate_max,
)
from src.simulations import (
    BaseExample,
    CustomUnconstrained,
    CustomUniform,
    SigmoidExample,
)
from src.test_data import best_versions

memory = Memory("joblib_cache")


rename_versions = {
    "19_bn": "-19 BN",
    "50": "-50",
    "11": "-11",
    "18": "-18",
    "152": "-152",
    "161": "-161",
    "121": "-121",
    "1_0": " 1.0",
    "0_5": " 0.5",
    "v3L": " V3L",
    "v2": " V2",
    "101": "-101",
    "b7": "-B7",
    "b0": "-B0",
    "l_16": " L-16",
    "b_16": " B-16",
    "large": " Large",
    "y_400mf": " y_400mf",
    "y_32gf": " y_32gf",
    "": "",
}


splits = [
    "test_c:_merged_no_rep5",
    "test_r",
    "test_c:snow5",
    "val",
]


# Figure 1
def test_fig1(out):
    """(Figure 1) Generate the 1D example accurate + calibrated."""
    fig = plot_fig_counter_example()
    save_fig(fig, out, pad_inches=0)


# Figure 2
def test_fig2(out, legend_right=True):
    """(Figure 2) Generate the intuition figure."""
    plt.rc("legend", fontsize=13)
    plt.rc("legend", columnspacing=0.4)
    plt.rc("legend", borderpad=0.3)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", labelspacing=0.3)
    fig = plot_fig_theorem_v2(
        isoline_right=False, squared=False, legend_right=legend_right
    )
    save_fig(fig, out, legend_right=legend_right, pad_inches=0)


# Figure 3
def test_fig3(out):
    """(Figure 3) Generate GL_induced figure."""
    fig = plot_fig_binning()
    save_fig(fig, out, pad_inches=0)


# Figure 4
@pytest.mark.parametrize(
    "grid",
    [
        list(product([15], np.linspace(1, 150, 50))),  # Clusters
        list(product(range(1, 31), [30])),  # Bins
    ],
)
def test_fig4(
    inp,
    out,
    n_jobs,
    grid,
    nocache,
):
    """(Figure 4) Generate the simulation figure."""
    calibrate = None
    clustering = "decision_tree"
    n_trials = 100
    n = 10000
    test_size = 0.5
    grid = list(grid)

    if len(np.unique([x[0] for x in grid])) == 1:
        which = "cluster"
    elif len(np.unique([x[1] for x in grid])) == 1:
        which = "bin"
    else:
        raise ValueError(
            f"grid must have an unique value for the first or "
            f"second axis. Got {grid}."
        )

    w = np.ones(2)
    w_perp = create_orthonormal_vector(w)
    ex = SigmoidExample(w, w_perp, bayes_opt=False, delta_width=1)

    assert isinstance(ex, BaseExample)

    GL = ex.GL_emp(N=10000000)

    kwargs = dict(
        clustering=clustering,
        trials=n_trials,
        which=which,
        ex=repr(ex),
        n=n,
    )

    path = save_path(inp, ext="csv", **kwargs)
    if nocache or not os.path.exists(path):
        rows = []

        def compute_one(trial, n_bins, n_samples_per_cluster_per_bin, strategy):
            n_samples_per_cluster_per_bin = int(n_samples_per_cluster_per_bin)
            n_bins = int(n_bins)
            Xt, y_labels = ex.generate_X_y(n=n, random_state=trial)
            y_scores = ex.S(Xt)

            if y_scores.ndim > 1:
                y_pred_scores, y_well_guess = binarize_multiclass_max(
                    y_scores, y_labels
                )
            else:
                y_pred_scores = y_scores
                y_well_guess = y_labels

            if strategy == "quantile":
                quantiles = np.linspace(0, 1, n_bins + 1)
                bins = np.percentile(y_pred_scores, quantiles * 100)
            elif strategy == "uniform":
                bins = np.linspace(0, 1, n_bins + 1)
            else:
                raise ValueError(f"Unknown strategy {strategy}.")

            splitter = bin_train_test_split(
                y_pred_scores,
                test_size=test_size,
                n_splits=1,
                bins=bins,
                random_state=0,
            )
            _, test_idx = next(splitter)

            if calibrate is not None:
                y_scores, _ = calibrate_scores(
                    y_scores,
                    y_labels,
                    method=calibrate,
                    test_size=test_idx,
                    max_calibration=True,
                )
                y_labels = y_well_guess

            (frac_pos, counts, *_) = cluster_evaluate(
                Xt,
                y_labels,
                y_scores,
                bins=bins,
                clustering=clustering,
                test_size=test_idx,
                n_samples_per_cluster_per_bin=n_samples_per_cluster_per_bin,
                verbose=0,
                n_jobs=1,
                return_clustering=True,
            )

            est = CEstimator(y_pred_scores, y_well_guess)
            c_hat = est.c_hat()

            LB_debiased, bias = grouping_loss_lower_bound(
                frac_pos, counts, debiased=True, return_bias=True
            )
            LB_biased = LB_debiased + bias
            GL_ind = estimate_GL_induced(c_hat, y_pred_scores, bins)

            return {
                "n_samples_per_cluster_per_bin": n_samples_per_cluster_per_bin,
                "LB_debiased": LB_debiased,
                "LB_biased": LB_biased,
                "bias": bias,
                "GL_ind": GL_ind,
                "GL": GL,
                "n_samples_per_cluster": np.mean(counts, where=counts > 0),
                "n_size_one_clusters": np.sum(counts == 1),
                "n_nonzero_clusters": np.sum(counts > 0),
                "strategy": strategy,
                "trial": trial,
                "n_bins": n_bins,
            }

        rows = Parallel(n_jobs=n_jobs)(
            delayed(compute_one)(trial, n_bins, n_clusters, strategy)
            for trial, (n_bins, n_clusters), strategy in tqdm(
                list(product(range(n_trials), grid, ["uniform"]))
            )
        )

        df = pd.DataFrame(rows)
        os.makedirs(inp, exist_ok=True)
        df.to_csv(path)

    df = pd.read_csv(path)

    if which == "cluster":
        df = df.query("n_samples_per_cluster_per_bin <= 100")
        fig = plot_simu(df, x="n_samples_per_cluster_per_bin", legend=True)
        ax = fig.axes[0]
        ax.set(
            xlabel=r"Ratio $\frac{\mathrm{number~of~samples}}{\mathrm{number~of~clusters}}$ per bin"
        )
        ax.set(ylabel=None)
        ax.set_yticklabels([])
        xmin = df["n_samples_per_cluster_per_bin"].min()
        xmax = df["n_samples_per_cluster_per_bin"].max()
        ax.set_xlim((xmin, xmax))

    elif which == "bin":
        fig = plot_simu(df, x="n_bins", legend=False)
        ax = fig.axes[0]
        ax.set(xlabel="Number of bins")
        xmin = df["n_bins"].min()
        xmax = df["n_bins"].max()
        ax.set_xlim((xmin, xmax))

    ax.set_title(" ", fontsize=1)  # For both figures to have same height
    ax = fig.axes[0]

    ymax = float(f"{2*df['GL'][0]:.1g}")
    ax.set_ylim((0, ymax))

    save_fig(fig, out, **kwargs)

    kwargs["which"] = "bin"
    path1 = save_path(inp, ext="csv", **kwargs)
    kwargs["which"] = "cluster"
    path2 = save_path(inp, ext="csv", **kwargs)
    kwargs["which"] = "both"

    if os.path.exists(path1) and os.path.exists(path2):
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", handlelength=1.3)
        plt.rc("legend", handletextpad=0.4)
        plt.rc("axes", titlepad=4)

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(3, 1.5), gridspec_kw=dict(wspace=0.03)
        )

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        df2 = df2.query("n_samples_per_cluster_per_bin <= 100")
        plot_simu(df2, x="n_samples_per_cluster_per_bin", legend=False, ax=ax1)
        ax1.set(xlabel=r"$\frac{\mathrm{\#~samples}}{\mathrm{\#~regions}}$ per bin")
        ax2.set(ylabel=None)
        ax2.set_yticklabels([])
        xmin = df2["n_samples_per_cluster_per_bin"].min()
        xmax = df2["n_samples_per_cluster_per_bin"].max()
        ax1.set_xlim((xmin, xmax))

        plot_simu(df1, x="n_bins", legend=True, ax=ax2)
        ax2.set(xlabel=r"# bins")
        xmin = df1["n_bins"].min()
        xmax = df1["n_bins"].max()
        ax2.set_xlim((xmin, xmax))
        ax1.set_yticks([0, 0.005, 0.01])
        ax2.set_xticks([5, 15, 25])
        ax2.set_yticks([])

        # For the minor ticks, use no labels; default NullFormatter.
        ax2.xaxis.set_minor_locator(MultipleLocator(5))

        ymax = float(f"{2*df1['GL'][0]:.1g}")
        ax1.set_ylim((0, ymax))
        ax2.set_ylim((0, ymax))

        ax1.set(title="a.")
        ax2.set(title="b.")

        save_fig(fig, out, pad_inches=0, **kwargs)


# Figure 5
def test_fig5(out):
    """(Figure 5) Generate the explanatory figure on grouping diagrams."""
    n_bins = 10
    n_clusters = 2
    mean_scores = (np.arange(n_bins) + 0.5) / n_bins
    mean_scores = np.tile(mean_scores[:, None], (1, n_clusters))
    counts = np.full((n_bins, n_clusters), 50, dtype=float)

    def f(x, alpha):
        return alpha * x * (x - 1) + x

    x_up = mean_scores[:, 0]
    x_down = mean_scores[:, 0]

    mu_up = f(x_up, alpha=-0.3)
    mu_down = f(x_down, alpha=0.9)

    frac_pos = np.stack([mu_up, mu_down], axis=1)

    prob_bins = calibration_curve(frac_pos, counts, mean_scores, return_mean_bins=False)

    size_gradient = np.power(np.linspace(0, 100, n_bins), 1 / 2)
    size_gradient[0] = 1
    counts *= size_gradient[:, None]

    hist = True
    ci = "clopper"
    min_cluster_size = 14
    capsize = 3.5
    cluster_size = 50
    vary_cluster_size = False
    absolute_size_scale = (10, None)
    plot_cal_hist = False
    figsize = (2.2, 2.2)
    legend_n_sizes = 0
    plt.rc("legend", title_fontsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("legend", handletextpad=0.01)
    plt.rc("legend", columnspacing=0.02)
    plt.rc("legend", borderpad=0.3)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", handlelength=1.2)
    plt.rc("legend", labelspacing=0.1)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("axes", labelsize=10.5)
    annotation_size = 15

    xlabel = "Confidence score"
    ylabel = "Fraction of positives  "

    fig = plot_frac_pos_vs_scores(
        frac_pos,
        counts,
        mean_scores,
        y_scores=None,
        y_labels=None,
        ncol=1,
        legend_loc="upper left",
        bbox_to_anchor=(0, 1),
        title=None,
        xlim_margin=0.05,
        ylim_margin=0.05,
        min_cluster_size=min_cluster_size,
        hist=hist,
        ci=ci,
        legend_cluster_sizes=False,
        vary_cluster_size=vary_cluster_size,
        capsize=capsize,
        xlabel=None,
        ylabel=None,
        cluster_size=cluster_size,
        absolute_size_scale=absolute_size_scale,
        plot_cal_hist=plot_cal_hist,
        figsize=figsize,
        legend_n_sizes=legend_n_sizes,
        legend_sizes_only=True,
        legend_min_max=False,
        plot_first_last_bins=False,
        grid_space=0,
        legend_title="Sizes",
    )
    ax = fig.axes[0]
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "", "1"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "", "", "", "100\n(%)"])

    fig.axes[1].get_xaxis().set_visible(False)
    fig.axes[2].get_yaxis().set_visible(False)

    delta = 0.06
    ax.annotate(
        xlabel,
        xy=(0.5, -delta),
        xytext=(0.5, -delta),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    ax.annotate(
        ylabel,
        xy=(-delta, 0.5),
        xytext=(-delta, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=plt.rcParams["axes.labelsize"],
        rotation=90,
    )

    # Position for mu
    i = 4
    j = 0
    mu_x = mean_scores[i, j]
    mu_y = frac_pos[i, j]

    # Position for c
    i = 4
    j = 0
    c_x = mean_scores[i, j]
    c_y = prob_bins[i]

    # Position for p
    i = 4
    j = 0
    p_x = mean_scores[i, j]
    p_y = 1

    ax.annotate(
        r"$\hat{\mu}^{(s)}_j$",
        xy=(mu_x, mu_y),
        xycoords="data",
        xytext=(0.29, 0.55),
        textcoords="data",
        va="bottom",
        ha="right",
        fontsize=annotation_size,
        arrowprops=dict(
            arrowstyle="->",
            shrinkB=4,
            patchA=None,
            shrinkA=7,
            connectionstyle="arc3,rad=.22",
        ),
    )
    ax.annotate(
        r"$\hat{c}^{(s)}$",
        xy=(c_x, c_y),
        xycoords="data",
        xytext=(0.6, 0.10),
        textcoords="data",
        va="bottom",
        ha="left",
        fontsize=annotation_size,
        arrowprops=dict(arrowstyle="->", shrinkB=2, connectionstyle="arc3,rad=-.22"),
    )
    ax.annotate(
        r"$n^{(s)}$",
        xy=(p_x, p_y),
        xycoords="data",
        xytext=(0.33, 0.8),
        textcoords="data",
        va="bottom",
        ha="right",
        fontsize=annotation_size,
        arrowprops=dict(
            arrowstyle="->",
            shrinkB=1,
            patchA=None,
            shrinkA=10,
            connectionstyle="arc3,rad=.22",
        ),
    )
    ax.annotate(
        "s",
        xy=(p_x, 0),
        xycoords="data",
        xytext=(p_x, 0),
        textcoords="data",
        va="bottom",
        ha="center",
        fontsize=annotation_size,
    )

    save_fig(fig, out, pad_inches=0)


# Figure 6, and 15 to 27
def plot_grouping_diagram(net, version, split, calibrate, clustering, out, n_jobs):
    """Generate the grouping diagrams of vision networks."""
    if version == "large":
        best = True
    elif version == "small":
        best = False
    else:
        raise ValueError(f'Unknown version "{version}".')

    if clustering != "decision_tree" and split != "test_r":
        pytest.skip(f"We use kmeans only on imagenet-r: {clustering} on {split}")

    if clustering == "decision_tree":
        clustering_name = "dt"
    elif clustering == "kmeans":
        clustering_name = "km"
    else:
        raise ValueError(f"Unknown {clustering}")

    n_bins = 15
    test_size = 0.5
    max_clusters_bin = 2  # for decision_tree only
    n_clusters = 2  # for kmeans only
    hist = True
    ci = "clopper"
    min_cluster_size = 14
    capsize = 3.5
    breakout = False
    cluster_size = 30
    vary_cluster_size = False
    absolute_size_scale = (10, 1500)
    plot_cal_hist = False
    figsize = (2.2, 2.2)
    legend_n_sizes = 1
    _renditions = False
    plt.rc("legend", title_fontsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("legend", handletextpad=0.01)
    plt.rc("legend", columnspacing=0.02)
    plt.rc("legend", borderpad=0.3)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", handlelength=1.2)
    plt.rc("legend", labelspacing=0.1)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("axes", labelsize=10.5)

    if best:
        version = best_versions.get(net, None)
        if version is None:
            pytest.skip(f'No best version "{version}" found for net "{net}"')

        positions = {
            IMAGENET_VGG: (0, 1),
            IMAGENET_ResNet: (0, 0),
            IMAGENET_DenseNet: (0, 0),
            IMAGENET_ShuffleNet: (0, 0),
            IMAGENET_MobileNet: (0, 1),
            IMAGENET_ResNext: (0, 0),
            IMAGENET_WideResNet: (0, 0),
            IMAGENET_MNASNet: (0, 0),
            IMAGENET_EfficientNet: (1, 1),
            IMAGENET_RegNet: (1, 0),
            IMAGENET_VisionTransformer: (1, 0),
            IMAGENET_ConvNeXt: (1, 0),
        }
        plot_xlabel, plot_ylabel = positions[net]
        net = net(split=split, type=version)

    else:
        positions = {
            IMAGENET_AlexNet: (0, 1),
            IMAGENET_VGG: (0, 0),
            IMAGENET_ResNet: (0, 0),
            IMAGENET_DenseNet: (0, 0),
            IMAGENET_Inception: (0, 1),
            IMAGENET_GoogLeNet: (0, 0),
            IMAGENET_ShuffleNet: (0, 0),
            IMAGENET_MobileNet: (0, 0),
            IMAGENET_ResNext: (0, 1),
            IMAGENET_WideResNet: (0, 0),
            IMAGENET_MNASNet: (0, 0),
            IMAGENET_EfficientNet: (0, 0),
            IMAGENET_RegNet: (1, 1),
            IMAGENET_VisionTransformer: (1, 0),
            IMAGENET_ConvNeXt: (1, 0),
        }
        plot_xlabel, plot_ylabel = positions[net]
        net = net(split=split)

    plot_xlabel = True  # override
    dirpath = net.get_default_dirpath()
    Xt = torch.load(join(dirpath, "Xt.pt")).numpy()
    y_scores = torch.load(join(dirpath, "y_scores.pt")).numpy()
    y_labels = torch.load(join(dirpath, "y_labels.pt")).numpy()

    y_pred_scores, y_well_guess = binarize_multiclass_max(y_scores, y_labels)
    splitter = bin_train_test_split(
        y_pred_scores, test_size=test_size, n_splits=1, bins=n_bins, random_state=0
    )
    train_idx, test_idx = next(splitter)

    ds = net.get_dataset()

    if _renditions:
        renditions = [
            re.match(f".*/([a-z]*)_[0-9]*.jpg", s).group(1) for s, _ in ds.imgs
        ]
        renditions = np.array(renditions)
        paths = np.array([s for s, _ in ds.imgs])

        print(renditions)
        assert len(renditions) == len(y_labels)

    if calibrate is not None:
        y_scores, _ = calibrate_scores(
            y_scores,
            y_labels,
            method=calibrate,
            test_size=test_idx,
            max_calibration=True,
        )
        y_labels = y_well_guess

    (frac_pos, counts, mean_scores, labels, *_) = cluster_evaluate_max(
        Xt,
        y_labels,
        y_scores,
        breakout=breakout,
        bins=n_bins,
        clustering=clustering,
        test_size=test_idx,
        min_samples_leaf=None,
        max_clusters_bin=max_clusters_bin,
        n_clusters=n_clusters,
        verbose=1,
        n_jobs=n_jobs,
        binary_scores=calibrate is not None,
        return_clustering=True,
    )

    if _renditions:
        labels_test = labels[test_idx]
        renditions_test = renditions[test_idx]
        paths_test = paths[test_idx]
        y_labels_test = y_labels[test_idx]

        df = pd.DataFrame(
            {
                "rendition": renditions_test,
                "clusters": labels_test,
                "count": 1,
                "paths": paths_test,
                "labels": y_labels_test,
            }
        )
        df_count = df.pivot_table(
            index="clusters", columns="labels", values="count", aggfunc=np.sum
        )

        pd.set_option("display.max_columns", 20)
        pd.set_option("display.max_rows", 200)

        df_percent = 100 * df_count / df_count.sum(axis=0)
        df_percent = df_percent.round(0)

        df_count.transpose().to_csv(join(out, f"rendition_counts_{net}.csv"))
        df_percent.transpose().to_csv(join(out, f"rendition_percent_{net}.csv"))

        for index, subdf in df.groupby(["rendition", "clusters"]):
            r, c = index
            dirpath = join(out, f"renditions_{net}", r, str(int(c)))
            for p in subdf["paths"]:
                os.makedirs(dirpath, exist_ok=True)
                dest_path = join(dirpath, p.replace("/", "_"))

    xlabel = "Confidence score"
    ylabel = "Correct predictions"
    fig = plot_frac_pos_vs_scores(
        frac_pos,
        counts,
        mean_scores,
        y_scores=None,
        y_labels=None,
        ncol=1,
        legend_loc="upper left",
        bbox_to_anchor=(0, 1),
        title=None,
        xlim_margin=0.05,
        ylim_margin=0.05,
        min_cluster_size=min_cluster_size,
        hist=hist,
        ci=ci,
        legend_cluster_sizes=True,
        vary_cluster_size=vary_cluster_size,
        capsize=capsize,
        xlabel="Confidence score",
        ylabel="Correct predictions (%)",
        cluster_size=cluster_size,
        absolute_size_scale=absolute_size_scale,
        plot_cal_hist=plot_cal_hist,
        figsize=figsize,
        legend_n_sizes=legend_n_sizes,
        legend_sizes_only=True,
        legend_min_max=False,
        plot_first_last_bins=False,
        grid_space=0,
        legend_title="Sizes",
    )

    ax = fig.axes[0]
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    ax.set_xlabel(None)
    ax.set_ylabel(None)

    fig.axes[1].get_xaxis().set_visible(False)
    fig.axes[2].get_yaxis().set_visible(False)

    delta = 0.06
    if plot_xlabel:
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "", "1"])
        ax.annotate(
            xlabel,
            xy=(0.5, -delta),
            xytext=(0.5, -delta),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=plt.rcParams["axes.labelsize"],
        )
    else:
        ax.xaxis.set_ticklabels([])

    if plot_ylabel:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0", "", "", "", "100\n(%)"])
        ax.annotate(
            ylabel,
            xy=(-delta, 0.5),
            xytext=(-delta, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=plt.rcParams["axes.labelsize"],
            rotation=90,
        )
    else:
        ax.yaxis.set_ticklabels([])

    out_kwargs = {
        "net": str(net),
        "cal": str(calibrate),
        "clu": clustering_name,
    }

    order = ["net", "cal"]
    save_fig(fig, out, order=order, **out_kwargs, pad_inches=0.0)


# Figure 6
@pytest.mark.parametrize("calibrate", [None, "isotonic"])
@pytest.mark.parametrize(
    "net",
    [
        IMAGENET_ConvNeXt,
        IMAGENET_VisionTransformer,
    ],
)
def test_fig6(calibrate, net, out, n_jobs):
    plot_grouping_diagram(
        net, "small", "test_r", calibrate, "decision_tree", out, n_jobs
    )


def plot_comparison_fig(
    nets: List[BaseNet],
    splits: List[str],
    which: str,
    inp: str,
    out: str,
    n_jobs: int,
    nocache: bool,
    append_versions: bool = True,
):
    """Generate the comparison figures used by Figures 7 and 14,
    (ECE and grouping loss) for all vision networks on all datasets."""
    n_bins = 15
    clustering = "decision_tree"
    test_size = 0.5
    hist = True
    ci = "clopper"
    min_cluster_size = 10
    breakout = False
    min_samples_leaf = None
    max_clusters_bin = None
    n_clusters = None
    n_samples_per_cluster_per_bin = 30

    bins = np.linspace(0, 1, n_bins + 1)

    if which == "small":
        bests = [False]
    elif which == "best":
        bests = [True]
    elif which == "both":
        bests = [False, True]
    else:
        raise ValueError(f'"{which}" not known.')

    def compute_one(split):
        def compute_one_net(net, calibrate, best):
            print(split, net, calibrate, best)
            if best:
                version = best_versions.get(net, None)
                if version is None:
                    print(f'No best version "{version}" found for net "{net}"')
                    return pd.DataFrame()
                net = net(split=split, type=version)  # best version
            else:
                net = net(split=split)  # default version (ie smallest)
            dirpath = net.get_default_dirpath()
            Xt = torch.load(join(dirpath, "Xt.pt")).numpy()
            y_scores = torch.load(join(dirpath, "y_scores.pt")).numpy()
            y_labels = torch.load(join(dirpath, "y_labels.pt")).numpy()

            y_pred_scores, y_well_guess = binarize_multiclass_max(y_scores, y_labels)
            splitter = bin_train_test_split(
                y_pred_scores,
                test_size=test_size,
                n_splits=1,
                bins=n_bins,
                random_state=0,
            )
            _, test_idx = next(splitter)

            _y_scores, _y_labels = y_scores, y_labels
            if calibrate is not None:
                y_scores, _ = calibrate_scores(
                    y_scores,
                    y_labels,
                    method=calibrate,
                    test_size=test_idx,
                    max_calibration=True,
                )
                y_labels = y_well_guess

            (frac_pos, counts, *_) = cluster_evaluate_max(
                Xt,
                y_labels,
                y_scores,
                breakout=breakout,
                bins=n_bins,
                verbose=2,
                n_jobs=4,
                min_samples_leaf=min_samples_leaf,
                max_clusters_bin=max_clusters_bin,
                n_samples_per_cluster_per_bin=n_samples_per_cluster_per_bin,
                clustering=clustering,
                n_clusters=n_clusters,
                test_size=test_idx,
                binary_scores=calibrate is not None,
            )

            if calibrate is None:
                y_pred_scores, y_well_guess = binarize_multiclass_max(
                    _y_scores, _y_labels
                )

            else:
                y_well_guess, y_pred_scores = y_labels[test_idx], y_scores[test_idx, 1]

            extra_out_kwargs = {
                "split": split,
                "dataset": net.get_dataset_name(),
                "network": net.get_class_name(False),
                "network+version": net.get_class_name(True),
                "calibrate": str(calibrate),
                "best": best,
                "n_samples_per_cluster_per_bin": n_samples_per_cluster_per_bin,
            }
            name = str(net)
            order = []
            out_kwargs, order = _get_out_kwargs(
                clustering,
                n_bins,
                ci,
                name,
                hist,
                test_size,
                calibrate,
                max_clusters_bin,
                min_samples_leaf,
                n_clusters,
                min_cluster_size,
                extra_out_kwargs,
                order,
            )
            out_kwargs["breakout"] = breakout

            metrics = {}
            metrics.update(out_kwargs)
            metrics.update(compute_multi_classif_metrics(_y_scores, _y_labels))
            metrics_binarized = compute_classif_metrics(y_pred_scores, y_well_guess)
            metrics_binarized.update(
                compute_calib_metrics(
                    frac_pos, counts, y_pred_scores, y_well_guess, bins
                )
            )
            metrics_binarized = {
                f"binarized_{k}": v for k, v in metrics_binarized.items()
            }
            metrics.update(metrics_binarized)

            df = pd.DataFrame([metrics])

            return df

        dfs = Parallel(n_jobs=n_jobs)(
            delayed(compute_one_net)(net, calibrate, best)
            for net, calibrate, best in tqdm(
                list(itertools.product(nets, [None, "isotonic"], bests))
            )
        )

        df = pd.concat(dfs, axis=0, ignore_index=True)
        return df

    os.makedirs(inp, exist_ok=True)
    df_paths = {
        split: join(inp, f"metrics_{split}_nspcb{n_samples_per_cluster_per_bin}.csv")
        for split in splits
    }
    for split, path in df_paths.items():
        print(split)
        if nocache or not os.path.exists(path):
            df = compute_one(split)
            df.to_csv(path)

        df = pd.read_csv(path, index_col=0)

        df_cal = df.query(f'calibrate == "isotonic"')
        df_ncal = df.query(f"calibrate.isna()")

        ece1 = df_ncal["binarized_msce"]
        ece2 = df_cal["binarized_msce"]
        glexp1 = df_ncal["binarized_lower_bound_debiased"]
        glexp2 = df_cal["binarized_lower_bound_debiased"]
        glexpbias1 = df_ncal["binarized_lower_bound_bias"]
        glexpbias2 = df_cal["binarized_lower_bound_bias"]
        glind1 = df_ncal["binarized_GL_ind"]
        glind2 = df_cal["binarized_GL_ind"]
        clind1 = df_ncal["binarized_CL_ind"]
        clind2 = df_cal["binarized_CL_ind"]
        acc1 = df_ncal["acc"]
        acc2 = df_cal["acc"]
        net_names1 = df_ncal["network+version"]
        net_names2 = df_cal["network+version"]

        assert list(net_names1) == list(net_names2)
        assert list(acc1) == list(acc2)

        names, versions = zip(*[(n.split(":") + [""])[:2] for n in net_names1])
        networks_by_name = {c.__name__.lower(): c for c in ALL_IMAGENET_NETWORKS}

        rename_names = {n: networks_by_name[n].__name__ for n in names}

        rename_names.update(
            {
                "wideresnet": "Wide ResNet",
                "resnext": "ResNeXt",
                "visiontransformer": "ViT",
            }
        )

        if append_versions:
            names = [
                f'{rename_names[n]}{rename_versions.get(v, " "+v.capitalize())}'
                for n, v in zip(names, versions)
            ]
        else:
            names = [f"{rename_names[n]}" for n, v in zip(names, versions)]

        splits = None
        figsize = (3.5, 6) if which == "both" else (3.6, 3.4)
        val1 = (ece1, glexp1, glexpbias1, glind1, clind1)
        val2 = (ece2, glexp2, glexpbias2, glind2, clind2)
        fig = barplots_ece_gl_cal(
            names,
            val1,
            val2,
            acc1,
            figsize=figsize,
            loc="lower right",
            bbox_to_anchor=(1, 0),
        )
        save_fig(
            fig,
            out,
            n="ece_gl",
            split=split,
            which=which,
            nspcb=n_samples_per_cluster_per_bin,
            pad_inches=0,
        )


# Figures 7
def test_fig7(inp, out, n_jobs, nocache, append_versions=True):
    plot_comparison_fig(
        nets=ALL_IMAGENET_NETWORKS,
        splits=["test_r"],
        which="small",
        inp=inp,
        out=out,
        n_jobs=n_jobs,
        nocache=nocache,
        append_versions=append_versions,
    )


# Figure 8
@pytest.mark.parametrize(
    "split",
    [
        "test_unseen",
        "test_seen",
    ],
)
@pytest.mark.parametrize(
    "calibrate",
    [
        None,
        "isotonic",
    ],
)
def test_fig8(calibrate, out, n_jobs, split):
    """(Figure 8) Generate the grouping diagrams of the NLP networks on YahooAnswers."""
    n_bins = 15
    clustering = "decision_tree"
    test_size = 0.5
    max_clusters_bin = 2
    hist = True
    ci = "clopper"
    min_cluster_size = 10
    capsize = 3.5
    cluster_size = 30
    vary_cluster_size = False
    absolute_size_scale = (10, None)
    plot_cal_hist = False
    figsize = (2.2, 2.2)
    legend_n_sizes = 1
    plt.rc("legend", title_fontsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("legend", handletextpad=0.01)
    plt.rc("legend", columnspacing=0.02)
    plt.rc("legend", borderpad=0.3)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", handlelength=1.2)
    plt.rc("legend", labelspacing=0.1)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("axes", labelsize=13)

    net = ZeroShotBartYahoo(split=split)
    dirpath = net.get_default_dirpath()
    Xt = torch.load(join(dirpath, "Xt.pt")).numpy()
    y_scores = torch.load(join(dirpath, "y_scores.pt")).numpy()
    y_labels = torch.load(join(dirpath, "y_labels.pt")).numpy()

    k = 1  # Plot the positive class

    splitter = bin_train_test_split(
        y_scores[:, k], test_size=test_size, n_splits=1, bins=n_bins, random_state=0
    )
    _, test_idx = next(splitter)

    if calibrate is not None:
        y_scores, _ = calibrate_scores(
            y_scores,
            y_labels,
            method=calibrate,
            test_size=test_idx,
            max_calibration=False,
        )

    class_name = net.get_class_names()[k]
    (frac_pos, counts, mean_scores, *_) = cluster_evaluate_marginals(
        Xt,
        y_labels,
        y_scores,
        positive_class=k,
        bins=n_bins,
        clustering=clustering,
        test_size=test_idx,
        min_samples_leaf=None,
        max_clusters_bin=max_clusters_bin,
        verbose=2,
        n_jobs=n_jobs,
    )

    fig = plot_frac_pos_vs_scores(
        frac_pos,
        counts,
        mean_scores,
        y_scores=None,
        y_labels=None,
        ncol=1,
        legend_loc="upper left",
        bbox_to_anchor=(0, 1),
        title=None,
        xlim_margin=0.05,
        ylim_margin=0.05,
        min_cluster_size=min_cluster_size,
        hist=hist,
        ci=ci,
        legend_cluster_sizes=True,
        vary_cluster_size=vary_cluster_size,
        capsize=capsize,
        xlabel="Confidence score",
        ylabel="Fraction of positives (%)",
        cluster_size=cluster_size,
        absolute_size_scale=absolute_size_scale,
        plot_cal_hist=plot_cal_hist,
        figsize=figsize,
        legend_n_sizes=legend_n_sizes,
        legend_sizes_only=True,
        legend_min_max=False,
        plot_first_last_bins=False,
        grid_space=0,
        legend_title="Sizes",
    )

    ax = fig.axes[0]
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])

    fig.axes[1].get_xaxis().set_visible(False)
    fig.axes[2].get_yaxis().set_visible(False)

    if calibrate is not None or not net.split == "test_seen":
        ax.set_ylabel(None)
        ax.yaxis.set_ticklabels([])

    out_kwargs = {
        "net": str(net),
        "cal": calibrate,
        "pos": class_name,
    }

    order = ["net", "cal"]
    save_fig(fig, out, order=order, **out_kwargs)


def plot_example_1d(name, out):
    """Generate the examples of calibrated classifiers (Figures 9 and 10)."""
    n = 1000000
    figsize = (2, 2.2)

    if name in ["poly", "2x", "step4"]:
        ex = CustomUniform(name=name, dist="gaussian")
        max_samples = 1000
    else:
        ex = CustomUnconstrained(name=name, x_min=-3, x_max=3)
        max_samples = 5000

    plt.rc("legend", fontsize=10)
    plt.rc("legend", title_fontsize=12)
    plt.rc("legend", handletextpad=0.5)
    plt.rc("legend", columnspacing=1.3)
    plt.rc("legend", borderpad=0.2)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", handlelength=1)
    plt.rc("legend", labelspacing=0.1)
    plt.rc("xtick", labelsize=9)
    plt.rc("ytick", labelsize=9)
    plt.rc("axes", labelsize=12)

    m = 100
    x_min = -2
    x_max = 2
    XX = np.linspace(x_min, x_max, m)
    Q = ex.f_star(XX)
    S = ex.f(XX)
    P = ex.p(XX)
    fig = plot_ffstar_1d(S, Q, P, x_min=x_min, x_max=x_max, figsize=figsize, lw=1.5)

    ax = fig.axes[1]
    ax.set_xticks([-2, 2])
    ax.set_xticklabels(["$-2$", "2"])
    delta = 0.04
    ax.set_xlabel(None)
    ax.annotate(
        "$X$",
        xy=(0.5, -delta),
        xytext=(0.5, -delta),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    save_fig(fig, out, link=name, n="X", order=["link", "n"], pad_inches=0.02)

    plt.rc("legend", title_fontsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("legend", handletextpad=0.01)
    plt.rc("legend", columnspacing=0.02)
    plt.rc("legend", borderpad=0.2)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", handlelength=0.8)
    plt.rc("legend", labelspacing=0.1)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("axes", labelsize=10.5)

    X, y_labels = ex.generate_X_y(n=n)
    y_scores = ex.f(X)
    y_true_probas = ex.f_star(X)
    fig = plot_score_vs_probas2(
        y_scores,
        y_labels,
        y_true_probas,
        max_samples=max_samples,
        height=2.5,
        grid_space=0,
        lim_margin=0.03,
        ncol=1,
        plot_first_last_bins=False,
    )

    ax = fig.axes[0]

    xlabel = "Confidence score $S$"
    ylabel = "True probability $Q$"
    plot_xlabel = True
    plot_ylabel = True

    ax.set_xlabel(None)
    ax.set_ylabel(None)

    delta = 0.04
    if plot_xlabel:
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "", "1"])
        ax.annotate(
            xlabel,
            xy=(0.5, -delta),
            xytext=(0.5, -delta),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=plt.rcParams["axes.labelsize"],
        )
    else:
        ax.xaxis.set_ticklabels([])

    delta = 0.02
    if plot_ylabel:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0", "", "", "", "1"])
        ax.annotate(
            ylabel,
            xy=(-delta, 0.5),
            xytext=(-delta, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=plt.rcParams["axes.labelsize"],
            rotation=90,
        )
    else:
        ax.yaxis.set_ticklabels([])
    save_fig(fig, out, link=name, n="QS", order=["link", "n"], pad_inches=0.02)


@pytest.mark.parametrize(
    "name",
    [
        "poly",
        "2x",
    ],
)
def test_fig9(name, out):
    plot_example_1d(name, out)


@pytest.mark.parametrize(
    "name",
    [
        "step4",
        "constant",
    ],
)
def test_fig10(name, out):
    plot_example_1d(name, out)


def plot_example_2d(bayes_opt, delta_width, out):
    """Generate the examples of calibrated classifiers
    on logistic regression (Figures 11 and 12) ."""
    plt.rc("legend", fontsize=10)
    plt.rc("legend", title_fontsize=12)
    plt.rc("legend", handletextpad=0.5)
    plt.rc("legend", columnspacing=1)
    plt.rc("legend", borderpad=0.2)
    plt.rc("legend", borderaxespad=0.1)
    plt.rc("legend", handlelength=1.6)
    plt.rc("legend", labelspacing=0.1)
    plt.rc("xtick", labelsize=9)
    plt.rc("ytick", labelsize=9)
    plt.rc("axes", labelsize=12)
    plt.rc("axes", titlesize=10)

    random_state = 0
    d = 2
    figsize = (2.2, 2.2)
    max_samples = 1000
    n = 1000000

    rng = check_random_state(random_state)

    w = rng.uniform(size=d)
    w /= np.linalg.norm(w)

    w_perp = create_orthonormal_vector(w)

    ex = SigmoidExample(w, w_perp, bayes_opt=bayes_opt, delta_width=delta_width)

    (
        fig1,
        fig2,
        fig3,
        fig4,
    ) = plot_ffstar_2d_v2(
        ex.f,
        ex.f_1d,
        ex.psi,
        ex.delta,
        ex.delta_max,
        w,
        w_perp,
        ex.mean,
        ex.cov,
        trim=True,
        figsize=figsize,
    )

    ax2 = fig2.axes[1]
    if bayes_opt:
        ax2.set_yticklabels([r"$-\frac{1}{4}$", "0", r"$\frac{1}{4}$"])
    else:
        ax2.set_yticklabels([r"$-\frac{1}{2}$", "0", r"$\frac{1}{2}$"])

    save_fig(
        fig1,
        out,
        f=1,
        bayes_opt=bayes_opt,
        delta_width=delta_width,
        order=["delta_width"],
        pad_inches=0.01,
    )
    save_fig(
        fig2,
        out,
        f=2,
        bayes_opt=bayes_opt,
        delta_width=delta_width,
        order=["delta_width"],
        pad_inches=0.01,
    )
    save_fig(
        fig3,
        out,
        f=3,
        bayes_opt=bayes_opt,
        delta_width=delta_width,
        order=["delta_width"],
        pad_inches=0.01,
    )

    plt.rc("legend", title_fontsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("legend", handletextpad=0.01)
    plt.rc("legend", columnspacing=0.02)
    plt.rc("legend", borderpad=0.2)
    plt.rc("legend", borderaxespad=0.2)
    plt.rc("legend", handlelength=0.8)
    plt.rc("legend", labelspacing=0.1)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("axes", labelsize=10.5)

    X, y_labels = ex.generate_X_y(n=n)
    y_scores = ex.f(X)
    y_true_probas = ex.f_star(X)
    fig = plot_score_vs_probas2(
        y_scores,
        y_labels,
        y_true_probas,
        max_samples=max_samples,
        height=2.5,
        grid_space=0,
        lim_margin=0.03,
        ncol=1,
        plot_first_last_bins=False,
    )

    ax = fig.axes[0]

    xlabel = "Confidence score $S$"
    ylabel = "True probability $Q$"
    plot_xlabel = True
    plot_ylabel = True

    ax.set_xlabel(None)
    ax.set_ylabel(None)

    delta = 0.04
    if plot_xlabel:
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "", "1"])
        ax.annotate(
            xlabel,
            xy=(0.5, -delta),
            xytext=(0.5, -delta),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=plt.rcParams["axes.labelsize"],
        )
    else:
        ax.xaxis.set_ticklabels([])

    delta = 0.02
    if plot_ylabel:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0", "", "", "", "1"])
        ax.annotate(
            ylabel,
            xy=(-delta, 0.5),
            xytext=(-delta, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=plt.rcParams["axes.labelsize"],
            rotation=90,
        )
    else:
        ax.yaxis.set_ticklabels([])
    save_fig(
        fig,
        out,
        link="",
        bayes_opt=bayes_opt,
        n="QS",
        delta_width=delta_width,
        order=["delta_width", "bayes_opt", "link", "n"],
        pad_inches=0.02,
    )


@pytest.mark.parametrize(
    "delta_width",
    [
        3,
        None,
    ],
)
def test_fig11(delta_width, out):
    plot_example_2d(bayes_opt=False, delta_width=delta_width, out=out)


@pytest.mark.parametrize(
    "delta_width",
    [
        3,
        None,
    ],
)
def test_fig12(delta_width, out):
    plot_example_2d(bayes_opt=True, delta_width=delta_width, out=out)


# Figure 13
def test_fig13(out, n_jobs):
    """(Figure 13) Generate the figures on renditions on ImageNet-R."""
    agg = "weighted_common"
    n_bins = 15
    max_clusters_bin = 2
    renditions = [
        "art",
        "cartoon",
        "graffiti",
        "embroidery",
        "graphic",
        "origami",
        "painting",
        "sculpture",
        "tattoo",
        "toy",
        "deviantart",
        "misc",
        "videogame",
        "sketch",
        "sticker",
    ]
    renditions = sorted(renditions)
    n_renditions = len(renditions)

    @memory.cache()
    def compute_one(net, with_type=False, agg="mean"):
        net = net(split="test_r")
        dirpath = net.get_default_dirpath()
        ds = net.get_dataset()

        Xt = torch.load(join(dirpath, "Xt.pt")).numpy()
        y_labels = torch.load(join(dirpath, "y_labels.pt")).numpy()
        y_scores = torch.load(join(dirpath, "y_scores.pt")).numpy()

        res = []
        for rendition in renditions:
            print(rendition)
            split = f"test_r:{rendition}"
            selection = np.array(
                [bool(re.match(f".*/{rendition}_.*", s)) for s, _ in ds.imgs]
            )
            net.split = split

            Xt_r = Xt[selection, :]
            y_scores_r = y_scores[selection, :]
            y_labels_r = y_labels[selection]

            (
                frac_pos,
                counts,
                mean_scores,
                *_,
            ) = cluster_evaluate_max(
                Xt_r,
                y_labels_r,
                y_scores_r,
                breakout=False,
                bins=n_bins,
                clustering=None,
                test_size=None,
                min_samples_leaf=None,
                max_clusters_bin=max_clusters_bin,
                verbose=1,
                n_jobs=n_jobs,
                binary_scores=False,
                return_clustering=False,
            )

            res.append((frac_pos, counts, mean_scores))

        # Zip results from list of tuples to tuple of lists
        L_frac_pos, L_counts, L_mean_scores = list(zip(*res))

        frac_pos = np.concatenate(L_frac_pos, axis=1)
        counts = np.concatenate(L_counts, axis=1)
        mean_scores = np.concatenate(L_mean_scores, axis=1)

        prob_bins, _ = calibration_curve(frac_pos, counts, mean_scores)

        if agg == "weighted_each":
            diff = np.sum(counts * frac_pos, axis=0) / np.sum(counts, axis=0) - np.sum(
                np.sum(counts, axis=1) * prob_bins / np.sum(counts)
            )

        elif agg == "weighted_common":
            diff = np.sum(counts * (frac_pos - prob_bins[:, None]), axis=0) / np.sum(
                counts, axis=0
            )

        elif agg == "mean":
            diff = np.mean(frac_pos - prob_bins[:, None], axis=0)

        elif agg == "bin":
            diff = frac_pos - prob_bins[:, None]

            df = pd.DataFrame(
                {
                    "diff": diff.flatten(),
                    "rendition": np.tile(
                        [s.capitalize() for s in renditions], (n_bins, 1)
                    ).flatten(),
                    "bin": np.tile(np.arange(n_bins), (1, n_renditions)).flatten(),
                    "net": net.get_class_name(with_type=with_type),
                }
            )

            return df

        else:
            raise ValueError(f"Unknown {agg}")

        df = pd.DataFrame(
            {
                "diff": diff,
                "rendition": [s.capitalize() for s in renditions],
                "net": net.get_class_name(with_type=with_type),
            }
        )

        return df

    with_type = True
    dfs = []
    for net in ALL_IMAGENET_NETWORKS:
        df = compute_one(net, with_type=with_type, agg=agg)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # Rename networks and versions
    names, versions = zip(*[(n.split(":") + [""])[:2] for n in df["net"]])
    networks_by_name = {c.__name__.lower(): c for c in ALL_IMAGENET_NETWORKS}
    rename_names = {n: networks_by_name[n].__name__ for n in names}
    rename_names.update(
        {
            "wideresnet": "Wide ResNet",
            "resnext": "ResNeXt",
            "visiontransformer": "ViT",
        }
    )
    df["net"] = [
        f'{rename_names[n]}{rename_versions.get(v, " "+v.capitalize())}'
        for n, v in zip(names, versions)
    ]

    fig = plot_renditions_calibration(df, x="diff", y="rendition", hue="net")
    save_fig(fig, out, with_type=with_type, agg=agg, pad_inches=0.01)


# Figures 14
def test_fig14(inp, out, n_jobs, nocache, append_versions=True):
    plot_comparison_fig(
        nets=ALL_IMAGENET_NETWORKS,
        splits=splits,
        which="both",
        inp=inp,
        out=out,
        n_jobs=n_jobs,
        nocache=nocache,
        append_versions=append_versions,
    )


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig15(net, out, n_jobs):
    plot_grouping_diagram(net, "small", "test_r", None, "decision_tree", out, n_jobs)


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig16(net, out, n_jobs):
    plot_grouping_diagram(net, "small", "test_r", None, "kmeans", out, n_jobs)


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig17(net, out, n_jobs):
    plot_grouping_diagram(net, "large", "test_r", None, "decision_tree", out, n_jobs)


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig18(net, out, n_jobs):
    plot_grouping_diagram(
        net, "small", "test_r", "isotonic", "decision_tree", out, n_jobs
    )


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig19(net, out, n_jobs):
    plot_grouping_diagram(
        net, "large", "test_r", "isotonic", "decision_tree", out, n_jobs
    )


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig20(net, out, n_jobs):
    plot_grouping_diagram(
        net, "small", "test_c:_merged_no_rep5", None, "decision_tree", out, n_jobs
    )


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig21(net, out, n_jobs):
    plot_grouping_diagram(
        net, "large", "test_c:_merged_no_rep5", None, "decision_tree", out, n_jobs
    )


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig22(net, out, n_jobs):
    plot_grouping_diagram(
        net, "small", "test_c:_merged_no_rep5", "isotonic", "decision_tree", out, n_jobs
    )


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig23(net, out, n_jobs):
    plot_grouping_diagram(
        net, "large", "test_c:_merged_no_rep5", "isotonic", "decision_tree", out, n_jobs
    )


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig24(net, out, n_jobs):
    plot_grouping_diagram(net, "small", "val", None, "decision_tree", out, n_jobs)


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig25(net, out, n_jobs):
    plot_grouping_diagram(net, "large", "val", None, "decision_tree", out, n_jobs)


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig26(net, out, n_jobs):
    plot_grouping_diagram(net, "small", "val", "isotonic", "decision_tree", out, n_jobs)


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS)
def test_fig27(net, out, n_jobs):
    plot_grouping_diagram(net, "large", "val", "isotonic", "decision_tree", out, n_jobs)
