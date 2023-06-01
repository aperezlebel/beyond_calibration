import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, lists, one_of
from sklearn.calibration import calibration_curve as sklearn_calibration_curve

from src._utils import pad_arrays
from src.grouping_loss import (
    calibration_curve,
    compute_calib_metrics,
    grouping_loss_lower_bound,
)
from src.partitioning import (
    cluster_evaluate,
    cluster_evaluate_marginals,
    cluster_evaluate_max,
)


@pytest.mark.parametrize("clustering", [None, "kmeans", "decision_tree"])
@given(
    X=arrays(float, (5, 2), elements=floats(0, 1)),
    y_labels=arrays(float, 5, elements=integers(0, 1)),
    y_scores=arrays(float, 5, elements=one_of(floats(0, 1), just(1))),
)
@settings(max_examples=20)
def test_cluster_evaluate_overall_mean(clustering, X, y_labels, y_scores):
    frac_pos, counts, mean_scores, *_ = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=3,
        n_clusters=3,
        clustering=clustering,
        min_samples_leaf=1,
        max_clusters_bin=None,
        test_size=None,
    )

    n = y_scores.shape[0]
    assert np.sum(counts) == n
    assert np.allclose(np.sum(frac_pos * counts) / n, np.mean(y_labels))
    assert np.allclose(np.sum(mean_scores * counts) / n, np.mean(y_scores))


def test_cluster_assignments(test_size=0.1):
    n, d = 20, 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=n)
    res = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=3,
        n_clusters=2,
        clustering="kmeans",
        max_clusters_bin=2,
        test_size=test_size,
        return_clustering=True,
    )

    (
        frac_pos,
        counts,
        mean_scores,
        labels,
        frac_pos_train,
        counts_train,
        mean_scores_train,
        labels_train,
    ) = res

    assert np.logical_xor(np.isnan(labels), np.isnan(labels_train)).all()


@pytest.mark.parametrize("clustering", ["kmeans", "decision_tree"])
@pytest.mark.parametrize("min_samples_leaf", [1, 2, 5, 10])
@pytest.mark.parametrize("test_size", [0.1, 0.5])
def test_cluster_evaluate_test_size(clustering, min_samples_leaf, test_size):
    n, d = 20, 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=n)
    res = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=3,
        n_clusters=3,
        clustering=clustering,
        min_samples_leaf=min_samples_leaf,
        max_clusters_bin=None,
        test_size=test_size,
    )

    frac_pos, counts, mean_scores, frac_pos_train, counts_train, mean_scores_train = res

    assert np.sum(counts) + np.sum(counts_train) == n
    assert np.allclose(
        (np.sum(frac_pos * counts) + np.sum(frac_pos_train * counts_train)) / n,
        np.mean(y_labels),
    )
    assert np.allclose(
        (np.sum(mean_scores * counts) + np.sum(mean_scores_train * counts_train)) / n,
        np.mean(y_scores),
    )


@pytest.mark.parametrize("test_size", [None, 0.1, 0.5])
@pytest.mark.parametrize("n_clusters", [1, 2, 10])
def test_cluster_evaluate_manual(test_size, n_clusters):
    n, d = 20, 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=n)
    clustering = rs.randint(0, n_clusters, size=n)
    res = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=3,
        n_clusters=3,
        clustering=clustering,
        test_size=test_size,
    )

    if test_size is None:
        frac_pos, counts, mean_scores = res
        frac_pos_train = np.zeros_like(frac_pos)
        counts_train = np.zeros_like(counts)
        mean_scores_train = np.zeros_like(mean_scores)
    else:
        (
            frac_pos,
            counts,
            mean_scores,
            frac_pos_train,
            counts_train,
            mean_scores_train,
        ) = res

    assert np.sum(counts) + np.sum(counts_train) == n
    assert np.allclose(
        (np.sum(frac_pos * counts) + np.sum(frac_pos_train * counts_train)) / n,
        np.mean(y_labels),
    )
    assert np.allclose(
        (np.sum(mean_scores * counts) + np.sum(mean_scores_train * counts_train)) / n,
        np.mean(y_scores),
    )


@pytest.mark.parametrize("clustering", ["kmeans", "decision_tree"])
@pytest.mark.parametrize("min_samples_leaf", [1, 2, 5, 10])
def test_cluster_evaluate_bins(clustering, min_samples_leaf):
    n, d = 10, 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=n)
    bins = 3
    frac_pos1, counts1, mean_scores1, *_ = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=bins,
        n_clusters=3,
        clustering=clustering,
        min_samples_leaf=min_samples_leaf,
        max_clusters_bin=None,
    )
    bins = np.linspace(0, 1, bins + 1)
    frac_pos2, counts2, mean_scores2, *_ = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=bins,
        n_clusters=3,
        clustering=clustering,
        min_samples_leaf=min_samples_leaf,
        max_clusters_bin=None,
    )
    assert np.allclose(frac_pos1, frac_pos2)
    assert np.allclose(counts1, counts2)
    assert np.allclose(mean_scores1, mean_scores2)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("n_clusters", [1, 2, 10])
@pytest.mark.parametrize("n_bins", [1, 2, 3, 10, 20])
@pytest.mark.parametrize("clustering", ["kmeans", "decision_tree"])
@pytest.mark.parametrize("min_samples_leaf", [1, 2, 5, 10])
def test_contains_calibration_curve(
    n, n_clusters, n_bins, clustering, min_samples_leaf
):
    d = 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=n)

    # Use calibration_curve of sklearn
    prob_true, prob_pred = sklearn_calibration_curve(y_labels, y_scores, n_bins=n_bins)

    # Compute calibration curve from cluster_evaluate results
    frac_pos, counts, mean_scores, *_ = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=n_bins,
        n_clusters=n_clusters,
        clustering=clustering,
        min_samples_leaf=min_samples_leaf,
        max_clusters_bin=None,
        test_size=None,
    )
    prob_true2, prob_pred2 = calibration_curve(
        frac_pos, counts, mean_scores, return_mean_bins=True
    )
    prob_true3 = calibration_curve(
        frac_pos, counts, mean_scores, return_mean_bins=False
    )

    hist = np.histogram(y_scores, bins=n_bins, range=(0, 1))[0]
    assert np.allclose(prob_true, prob_true2)
    assert np.allclose(prob_pred, prob_pred2)
    assert np.array_equal(prob_true2, prob_true3)
    assert np.allclose(np.sum(counts, axis=1), hist)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("n_clusters", [1, 2, 10])
@pytest.mark.parametrize("n_bins", [1, 2, 10])
@pytest.mark.parametrize("clustering", ["kmeans", "decision_tree"])
def test_cluster_evaluate_marginals_all_classes(
    n, K, n_clusters, n_bins, n_jobs, clustering
):
    """Test if cluster_evaluate_marginals with positive_class=None is the same
    as concatenating results from cluster_evaluate_marginals with
    positive_class=k for k in range(K)."""
    d = 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=(n, K))

    # Compute calibration curve from cluster_evaluate results
    (frac_pos1, counts1, mean_scores1) = cluster_evaluate_marginals(
        X,
        y_labels,
        y_scores,
        positive_class=None,
        bins=n_bins,
        n_clusters=n_clusters,
        verbose=0,
        n_jobs=n_jobs,
        clustering=clustering,
    )
    L_frac_pos = []
    L_counts = []
    L_mean_scores = []

    for positive_class in range(K):
        (frac_pos2, counts2, mean_scores2) = cluster_evaluate_marginals(
            X,
            y_labels,
            y_scores,
            positive_class=positive_class,
            bins=n_bins,
            n_clusters=n_clusters,
            clustering=clustering,
            verbose=0,
        )

        L_frac_pos.append(frac_pos2)
        L_counts.append(counts2)
        L_mean_scores.append(mean_scores2)

    L_frac_pos = pad_arrays(L_frac_pos)
    L_counts = pad_arrays(L_counts)
    L_mean_scores = pad_arrays(L_mean_scores)

    frac_pos2 = np.stack(L_frac_pos, axis=2)
    counts2 = np.stack(L_counts, axis=2)
    mean_scores2 = np.stack(L_mean_scores, axis=2)

    assert np.allclose(frac_pos1, frac_pos2)
    assert np.allclose(counts1, counts2)
    assert np.allclose(mean_scores1, mean_scores2)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("n_clusters", [1, 2, 10])
@pytest.mark.parametrize("n_bins", [1, 2, 10])
@pytest.mark.parametrize("clustering", ["kmeans", "decision_tree"])
def test_calibration_curve_multiclass(n, K, n_bins, n_clusters, clustering):
    d = 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=(n, K))

    # Compute calibration curve from cluster_evaluate results
    (frac_pos, counts, mean_scores) = cluster_evaluate_marginals(
        X,
        y_labels,
        y_scores,
        positive_class=None,
        bins=n_bins,
        n_clusters=n_clusters,
        clustering=clustering,
        verbose=0,
    )

    assert frac_pos.shape == counts.shape
    assert mean_scores.shape == counts.shape

    prob_bins, mean_bins = calibration_curve(frac_pos, counts, mean_scores)

    assert prob_bins.shape == (n_bins, K)
    assert mean_bins.shape == (n_bins, K)

    for positive_class in range(K):
        y_labels_k = (y_labels == positive_class).astype(int)
        y_scores_k = y_scores[:, positive_class]

        # Use calibration_curve of sklearn
        prob_true_k1, prob_pred_k1 = sklearn_calibration_curve(
            y_labels_k, y_scores_k, n_bins=n_bins
        )

        prob_pred_k2 = mean_bins[:, positive_class]
        prob_true_k2 = prob_bins[:, positive_class]

        prob_pred_k2 = prob_pred_k2[~np.isnan(prob_pred_k2)]
        prob_true_k2 = prob_true_k2[~np.isnan(prob_true_k2)]

        assert np.allclose(prob_pred_k1, prob_pred_k2)
        assert np.allclose(prob_true_k1, prob_true_k2)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("n_clusters", [1, 2, 10])
@pytest.mark.parametrize("n_bins", [1, 2, 10])
@pytest.mark.parametrize("clustering", ["kmeans", "decision_tree"])
def test_cluster_evaluate_max_breakout(n, K, n_clusters, n_bins, clustering):
    """Test if cluster_evaluate_max with or without breakout gives the same
    calibration curve."""
    d = 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=(n, K))

    # Compute calibration curve from cluster_evaluate results
    (frac_pos1, counts1, mean_scores1, *_) = cluster_evaluate_max(
        X,
        y_labels,
        y_scores,
        breakout=True,
        bins=n_bins,
        n_clusters=n_clusters,
        clustering=clustering,
        verbose=0,
    )

    (frac_pos2, counts2, mean_scores2, *_) = cluster_evaluate_max(
        X,
        y_labels,
        y_scores,
        breakout=False,
        bins=n_bins,
        n_clusters=n_clusters,
        clustering=clustering,
        verbose=0,
    )

    prob_bins1, mean_bins1 = calibration_curve(frac_pos1, counts1, mean_scores1)
    prob_bins2, mean_bins2 = calibration_curve(frac_pos2, counts2, mean_scores2)

    assert np.allclose(prob_bins1, prob_bins2)
    assert np.allclose(mean_bins1, mean_bins2)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("n_clusters", [1, 2, 10])
@pytest.mark.parametrize("n_bins", [1, 2, 3, 10, 20])
@pytest.mark.parametrize("clustering", ["kmeans", "decision_tree"])
@pytest.mark.parametrize("min_samples_leaf", [1, 2, 5, 10])
def test_calib_metrics(n, n_clusters, n_bins, clustering, min_samples_leaf):
    d = 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=n)

    bins = np.linspace(0, 1, n_bins + 1)

    # Compute calibration curve from cluster_evaluate results
    frac_pos, counts, *_ = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=bins,
        n_clusters=n_clusters,
        clustering=clustering,
        min_samples_leaf=min_samples_leaf,
        max_clusters_bin=None,
    )

    metrics = compute_calib_metrics(frac_pos, counts, y_scores, y_labels, bins)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("n_clusters", [1, 2, 10])
@pytest.mark.parametrize("n_bins", [1, 2, 3, 10, 20])
@pytest.mark.parametrize("clustering", ["kmeans", "decision_tree"])
@pytest.mark.parametrize("min_samples_leaf", [1, 2, 5, 10])
def test_bounds(n, n_clusters, n_bins, clustering, min_samples_leaf):
    d = 2
    rs = np.random.RandomState(0)
    X = rs.uniform(size=(n, d))
    y_labels = rs.randint(0, 2, size=n)
    y_scores = rs.uniform(size=n)

    bins = np.linspace(0, 1, n_bins + 1)

    # Compute calibration curve from cluster_evaluate results
    frac_pos, counts, *_ = cluster_evaluate(
        X,
        y_labels,
        y_scores,
        bins=bins,
        n_clusters=n_clusters,
        clustering=clustering,
        min_samples_leaf=min_samples_leaf,
        max_clusters_bin=None,
        test_size=None,
    )

    lower_bound_bin = grouping_loss_lower_bound(frac_pos, counts, reduce_bin=False)
    lower_bound = grouping_loss_lower_bound(frac_pos, counts, reduce_bin=True)
    lower_bound2 = np.nansum(lower_bound_bin)

    assert np.allclose(lower_bound, lower_bound2)
