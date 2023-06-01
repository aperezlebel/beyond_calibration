"""Functions to estimate binning-induced grouping loss."""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from src._utils import _validate_clustering


def compute_calib_metrics(frac_pos, counts, y_scores, y_labels, bins):
    """Compute calibration metrics from output of clustering.

    Parameters
    ----------
    frac_pos : (n_bins, n_clusters) array
        The fraction of positives in each cluster for each bin.

    counts : (n_bins, n_clusters) array
        The number of samples in each cluster for each bin.

    mean_scores : (n_bins, n_clusters) array
        The mean score of samples in each cluster for each bin.

    y_scores : (n_samples,) or (n_samples, n_classes) array
        Array of classification scores. If 1D, binary classification
        with threhsold at 0.5 is assumed.

    y_labels : (n_samples,) array
        True labels, taking at most n_classes values (n_classes=2) if binary.

    bins : int or (n_bins+1,) array
        Number of equaly spaced bins or array defining the bins bounds.

    Returns
    -------
    metrics : dict

    """
    _validate_clustering(frac_pos, counts)

    try:
        bins = np.linspace(0, 1, bins + 1)
    except TypeError:
        pass

    lower_bound = grouping_loss_lower_bound(frac_pos, counts, reduce_bin=True)

    lower_bound_debiased, bias = grouping_loss_lower_bound(
        frac_pos, counts, reduce_bin=True, debiased=True, return_bias=True
    )

    # Estimation of GL_induced
    est = CEstimator(y_scores, y_labels)
    c_hat = est.c_hat()
    GL_ind = estimate_GL_induced(c_hat, y_scores, bins)
    CL_ind = estimate_CL_induced(c_hat, y_scores, bins)

    metrics = {
        "lower_bound": lower_bound,
        "lower_bound_debiased": lower_bound_debiased,
        "lower_bound_bias": bias,
        "n_samples_per_cluster": np.mean(counts, where=counts > 0),
        "n_size_one_clusters": np.sum(counts == 1),
        "n_nonzero_clusters": np.sum(counts > 0),
        "n_bins": len(bins) - 1,
        "GL_ind": GL_ind,
        "CL_ind": CL_ind,
        "GL_ind_est": "KNNRegressor(n_neighbors=2000)",
        "CL_ind_est": "KNNRegressor(n_neighbors=2000)",
    }

    return metrics


def calibration_curve(
    frac_pos, counts, mean_scores=None, remove_empty=True, return_mean_bins=True
):
    """Compute calibration curve from output of clustering.
    Result is the same as sklearn's calibration_curve.

    Parameters
    ----------
    frac_pos : (bins, n_clusters) array
        The fraction of positives in each cluster for each bin.

    counts : (bins, n_clusters) array
        The number of samples in each cluster for each bin.

    mean_scores : (bins, n_clusters) array
        The mean score of samples in each cluster for each bin.

    remove_empty : bool
        Whether to remove empty bins.

    return_mean_bins : bool
        Whether to return mean_bins.

    Returns
    -------
    prob_bins : (bins,) arrays
        Fraction of positives in each bin.

    mean_bins : (bins,) arrays
        Mean score in each bin. Returned only if return_mean_bins=True.

    """
    if not return_mean_bins:
        _validate_clustering(frac_pos, counts)

    else:
        _validate_clustering(frac_pos, counts, mean_scores)

    if return_mean_bins and mean_scores is None:
        raise ValueError("mean_scores cannot be None when " "return_mean_bins=True.")

    count_sums = np.sum(counts, axis=1, dtype=float)
    non_empty = count_sums > 0
    prob_bins = np.divide(
        np.sum(frac_pos * counts, axis=1),
        count_sums,
        where=non_empty,
        out=np.full_like(count_sums, np.nan),
    )

    if return_mean_bins:
        mean_bins = np.divide(
            np.sum(mean_scores * counts, axis=1),
            count_sums,
            where=non_empty,
            out=np.full_like(count_sums, np.nan),
        )

    # The calibration_curve of sklearn removes empty bins.
    # Should do the same to give same result.
    if frac_pos.ndim == 2 and remove_empty:
        prob_bins = prob_bins[non_empty]
        if return_mean_bins:
            mean_bins = mean_bins[non_empty]

    if return_mean_bins:
        return prob_bins, mean_bins

    return prob_bins


def grouping_loss_bias(frac_pos, counts, reduce_bin=True):
    prob_bins = calibration_curve(
        frac_pos, counts, remove_empty=False, return_mean_bins=False
    )
    n_bins = np.sum(counts, axis=1)  # number of samples in bin
    n = np.sum(n_bins)
    var = np.divide(
        frac_pos * (1 - frac_pos),
        counts - 1,
        np.full_like(frac_pos, np.nan, dtype=float),
        where=counts > 1,
    )
    var = var * np.divide(
        counts,
        n_bins[:, None],
        np.full_like(frac_pos, np.nan, dtype=float),
        where=n_bins[:, None] > 0,
    )
    bias = np.nansum(var, axis=1) - np.divide(prob_bins * (1 - prob_bins), n_bins - 1)
    bias *= n_bins / n
    if reduce_bin:
        return np.nansum(bias)

    return bias


def grouping_loss_lower_bound(
    frac_pos,
    counts,
    reduce_bin=True,
    debiased=False,
    return_bias=False,
):
    """Compute a lower bound of the grouping loss from clustering."""
    prob_bins = calibration_curve(
        frac_pos, counts, remove_empty=False, return_mean_bins=False
    )
    diff = np.multiply(counts, np.square(frac_pos - prob_bins[:, None]))

    if reduce_bin:
        lower_bound = np.nansum(diff) / np.sum(counts)

    else:
        lower_bound = np.divide(np.nansum(diff, axis=1), np.sum(counts))

    if debiased:
        bias = np.nan_to_num(
            grouping_loss_bias(frac_pos, counts, reduce_bin=reduce_bin)
        )
        lower_bound -= bias
        if return_bias:
            return lower_bound, bias

    return lower_bound


def check_2D_array(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    elif x.ndim == 2 and x.shape[1] != 1:
        raise ValueError(f"x must have one feature. Got shape " f"{x.shape}")

    elif x.ndim > 2:
        raise ValueError(f"x must be at most 2 dimensional. " f"Got shape {x.shape}")

    return x


class CEstimator:
    def __init__(self, y_scores, y_labels):
        y_scores = np.array(y_scores)
        y_labels = np.array(y_labels)

        y_scores = check_2D_array(y_scores)

        self.y_scores = y_scores
        self.y_labels = y_labels

    def _c_hat(self, test_scores):
        test_scores = check_2D_array(test_scores)
        n_neighbors = min(2000, int(0.1 * len(test_scores)))
        est = KNeighborsRegressor(n_neighbors=n_neighbors)
        est.fit(self.y_scores.reshape(-1, 1), self.y_labels)
        c_hat = est.predict(test_scores)
        return c_hat

    def c_hat(self):
        return self._c_hat(self.y_scores.reshape(-1, 1))


def estimate_GL_induced(c_hat, y_scores, bins):
    """Estimate GL induced for the Brier score."""
    n_bins = len(bins) - 1
    y_bins = np.digitize(y_scores, bins=bins) - 1
    y_bins = np.clip(y_bins, a_min=None, a_max=n_bins - 1)

    uniques, counts = np.unique(y_bins, return_counts=True)
    var = []

    for i in uniques:
        var.append(np.var(c_hat[y_bins == i]))

    GL_ind = np.vdot(var, counts) / np.sum(counts)

    return GL_ind


def estimate_CL_induced(c_hat, y_scores, bins):
    """Estimate CL induced for the Brier score."""
    n_bins = len(bins) - 1
    y_bins = np.digitize(y_scores, bins=bins) - 1
    y_bins = np.clip(y_bins, a_min=None, a_max=n_bins - 1)

    uniques, counts = np.unique(y_bins, return_counts=True)
    var = []

    S_minus_C = y_scores - c_hat

    for i in uniques:
        var.append(np.var(S_minus_C[y_bins == i]))

    CL_ind = -np.vdot(var, counts) / np.sum(counts)

    return CL_ind
