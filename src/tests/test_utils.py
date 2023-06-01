import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis.strategies import integers, lists
from sklearn.metrics import pairwise_distances
from torch.nn import functional as F

from src._utils import calibrate_scores, pad_array, pad_arrays, pairwise_call, save_fig
from src.grouping_loss import grouping_loss_bias, grouping_loss_lower_bound
from src.partitioning import cluster_evaluate


@given(
    test_size=lists(integers(0, 19), unique=True),
)
def test_calibrate(test_size):
    n = 100
    K = 3
    rs = np.random.RandomState(0)
    y_scores = rs.uniform(-100, 100, size=(n, K))
    y_scores = np.array(F.softmax(torch.from_numpy(y_scores), dim=1))
    y_labels = rs.binomial(n=K - 1, p=0.5, size=n)

    Xt = np.zeros((n, 1))
    y_scores_max = np.max(y_scores, axis=1)
    y_labels_pred = np.argmax(y_scores, axis=1)

    y_labels_binarized = (y_labels_pred == y_labels).astype(int)
    cluster_evaluate(Xt, y_labels_binarized, y_scores_max, bins=15)

    y_scores_cal, test_idx = calibrate_scores(y_scores, y_labels, test_size=test_size)
    assert np.array_equal(np.sort(test_size), test_idx)

    y_scores_cal = y_scores_cal[test_idx]
    Xt = Xt[test_idx, :]
    y_labels_binarized = y_labels_binarized[test_idx]
    y_scores_max_cal = np.max(y_scores_cal, axis=1)
    cluster_evaluate(Xt, y_labels_binarized, y_scores_max_cal, bins=15)


@pytest.mark.parametrize("n", [10, 100])
def test_grouping_loss_bias(n):
    d = 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=n)
    n_bins = 10

    bins = np.linspace(0, 1, n_bins + 1)

    # Compute calibration curve from cluster_evaluate results
    frac_pos, counts, *_ = cluster_evaluate(X, y_labels, y_scores, bins=bins)

    bias_bin = grouping_loss_bias(frac_pos, counts, reduce_bin=False)
    bias = grouping_loss_bias(frac_pos, counts, reduce_bin=True)

    assert np.allclose(np.nansum(bias_bin), bias)

    lower_bound_bin = grouping_loss_lower_bound(
        frac_pos, counts, reduce_bin=False, debiased=True
    )
    lower_bound = grouping_loss_lower_bound(
        frac_pos, counts, reduce_bin=True, debiased=True
    )

    assert np.allclose(np.sum(lower_bound_bin), lower_bound)


@pytest.mark.parametrize("n", [10, 100])
def test_grouping_loss_bias_nans(n):
    d = 2

    frac_pos = np.full((3, 2), 0.5)
    counts = np.full_like(frac_pos, 2)
    counts[0, 0] = 1
    counts[2, :] = 0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bias_bin = grouping_loss_bias(frac_pos, counts, reduce_bin=False)
        bias = grouping_loss_bias(frac_pos, counts, reduce_bin=True)

    assert np.allclose(np.nansum(bias_bin), bias)

    lower_bound_bin = grouping_loss_lower_bound(
        frac_pos, counts, reduce_bin=False, debiased=True
    )
    lower_bound = grouping_loss_lower_bound(
        frac_pos, counts, reduce_bin=True, debiased=True
    )

    assert np.allclose(np.sum(lower_bound_bin), lower_bound)


def test_pad_array():
    a = np.array([1, 2])

    b = pad_array(a, (3,), fill_value=0)
    assert np.array_equal(b, [1, 2, 0])

    b = pad_array(a, a.shape, fill_value=0)
    assert np.array_equal(b, a)

    a = np.array([[1, 2], [3, 4]], dtype=float)

    b = pad_array(a, a.shape, fill_value=0)
    assert np.array_equal(b, a)

    b = pad_array(a, (3, 2), fill_value=np.nan)
    c = np.array([[1, 2], [3, 4], [None, None]], dtype=float)
    assert np.array_equal(b, c, equal_nan=True)


def test_pad_arrays():
    a1 = np.array([1, 2])
    a2 = np.array([1, 2, 3])
    L = [a1, a2]

    L2 = pad_arrays(L, shape=None)
    assert np.array_equal(L2[0].shape, L2[1].shape)


def test_save_fig():
    dir = tempfile.TemporaryDirectory()
    dirpath = dir.name

    fig = plt.figure()

    kwargs = {
        "arg3": "arg3",
        "arg1": 0,
        "arg2": True,
    }

    filepath = save_fig(fig, dirpath)
    assert os.path.basename(filepath) == "fig.pdf"
    filepath = save_fig(fig, dirpath, **kwargs)
    assert os.path.basename(filepath) == "arg1=0:arg2=T:arg3=arg3.pdf"
    filepath = save_fig(fig, dirpath, order=["arg2"], **kwargs)
    assert os.path.basename(filepath) == "arg2=T:arg1=0:arg3=arg3.pdf"
    filepath = save_fig(fig, dirpath, order=["arg2", "arg3"], **kwargs)
    assert os.path.basename(filepath) == "arg2=T:arg3=arg3:arg1=0.pdf"

    dir.cleanup()


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_pairwise_call(symmetric, n_jobs):
    n = 5
    d = 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    D1 = pairwise_distances(X, metric="euclidean")
    D2 = pairwise_call(
        X, lambda x, y: np.linalg.norm(x - y), symmetric=symmetric, n_jobs=n_jobs
    )
    assert np.allclose(D1, D2)
