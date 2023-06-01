"""Helper functions for the src module."""
import hashlib
import os

import numpy as np
import progressbar
import torch
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torchmetrics import AUROC, Accuracy, CalibrationError, MeanSquaredError
from tqdm import tqdm


def _validate_clustering(*args):
    if len(args) == 2:
        frac_pos, counts = args
    elif len(args) == 3:
        frac_pos, counts, mean_scores = args
    else:
        raise ValueError(f"2 or 3 args must be given. Got {len(args)}.")

    if frac_pos.shape != counts.shape:
        raise ValueError(
            f"Shape mismatch between frac_pos {frac_pos.shape} and counts {counts.shape}"
        )

    if len(args) == 3 and frac_pos.shape != mean_scores.shape:
        raise ValueError(
            f"Shape mismatch between frac_pos {frac_pos.shape} and mean_scores {mean_scores.shape}"
        )

    if frac_pos.ndim < 2:
        raise ValueError(
            f"frac_pos, counts and mean_scores must bet at least "
            f"2D. Got {frac_pos.ndim}D."
        )


def _validate_scores(y_scores, one_dim=False):
    if one_dim is not None and one_dim and y_scores.ndim != 1:
        raise ValueError(f"y_scores must be 1D. Got shape {y_scores.shape}.")

    if one_dim is not None and not one_dim and y_scores.ndim != 2:
        raise ValueError(f"y_scores must be 2D. Got shape {y_scores.shape}.")

    if one_dim is None and y_scores.ndim not in [1, 2]:
        raise ValueError(f"y_scores must be 1D or 2D. Got shape {y_scores.shape}.")

    if np.any(y_scores < 0) or np.any(y_scores > 1):
        raise ValueError("y_scores must take values between 0 and 1.")

    if y_scores.ndim == 2 and not np.allclose(np.sum(y_scores, axis=1), 1):
        raise ValueError("y_scores must sum to 1 class-wise when 2D.")


def _validate_labels(y_labels, binary=False):
    uniques = np.unique(y_labels)
    if binary and len(uniques) > 2:
        raise ValueError(f"y_labels must be binary. Found values: {uniques}.")


def scores_to_id_bins(y_scores, bins):
    y_bins = np.digitize(y_scores, bins=bins) - 1
    y_bins = np.clip(y_bins, 0, len(bins) - 2)
    return y_bins


def scores_to_pred(y_scores, threshold=0.5):
    _validate_scores(y_scores, one_dim=None)

    if y_scores.ndim == 1:
        y_pred = (y_scores >= threshold).astype(int)
        y_pred_scores = y_pred * y_scores + (1 - y_pred) * (1 - y_scores)

    elif y_scores.ndim == 2:
        y_pred = np.argmax(y_scores, axis=1).astype(int)
        y_pred_scores = np.max(y_scores, axis=1)

    return y_pred, y_pred_scores


def binarize_multiclass_max(y_scores, y_labels):
    _validate_scores(y_scores, one_dim=False)
    y_pred, y_pred_scores = scores_to_pred(y_scores)
    y_well_guess = (y_pred == y_labels).astype(int)
    return y_pred_scores, y_well_guess


def scores_to_calibrated_scores(y_scores, prob_bins, bins):
    _validate_scores(y_scores, one_dim=True)

    if len(prob_bins) != len(bins) - 1:
        raise ValueError(
            f"prob_bins must have {len(bins)-1} elements." f"Got {len(prob_bins)}."
        )

    y_bins = scores_to_id_bins(y_scores, bins)
    y_scores_cal = prob_bins[y_bins]
    return y_scores_cal


def brier_multi(y_scores, y_labels):
    y_labels = np.array(y_labels, dtype=int)
    y_binary = np.zeros_like(y_scores, dtype=int)
    y_binary[np.arange(len(y_labels)), y_labels] = 1
    return np.mean(np.sum(np.square(y_scores - y_binary), axis=1))


def compute_multi_classif_metrics(y_scores, y_labels):
    _validate_scores(y_scores, one_dim=None)
    if y_scores.ndim == 1:
        _y_scores = np.stack([1 - y_scores, y_scores], axis=1)
    else:
        _y_scores = y_scores

    y_pred, _ = scores_to_pred(y_scores)

    y_pred_scores, y_well_guess = binarize_multiclass_max(_y_scores, y_labels)

    y_well_guess = torch.from_numpy(y_well_guess)
    y_pred_scores = torch.from_numpy(y_pred_scores)

    if y_scores.ndim == 2 and y_scores.shape[1] == 2:
        auc = roc_auc_score(y_labels, y_scores[:, 1])
    else:
        try:
            auc = roc_auc_score(y_labels, y_scores, multi_class="ovr")
        except ValueError:
            auc = None

    metrics = {
        "acc": accuracy_score(y_labels, y_pred),
        "auc": auc,
        "brier_multi": brier_multi(y_scores, y_labels),
        "max_ece": CalibrationError(norm="l1", compute_on_step=True)
        .forward(y_pred_scores, y_well_guess)
        .item(),
        "max_mce": CalibrationError(norm="max", compute_on_step=True)
        .forward(y_pred_scores, y_well_guess)
        .item(),
        "max_rmsce": CalibrationError(norm="l2", compute_on_step=True)
        .forward(y_pred_scores, y_well_guess)
        .item(),
    }
    metrics["max_msce"] = np.square(metrics["max_rmsce"])

    return metrics


def bin_train_test_split(
    y_scores, test_size=0.5, n_splits=10, bins=15, random_state=0, stratss=False
):
    try:
        bins = np.linspace(0, 1, bins + 1)
    except TypeError:
        pass

    # Preserve proportion of scores of each bin in train and test
    n_samples = y_scores.shape[0]
    y_bins = scores_to_id_bins(y_scores, bins)
    if stratss:
        cv = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
        split = cv.split(np.zeros(n_samples), y_bins)

    else:

        def mysplit(y_scores, n_splits, test_size):
            indices = np.arange(y_scores.shape[0])
            cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
            shuffle_splits = []

            # Create one iterator in each bin
            for i in range(len(bins)):
                y_scores_bin = y_scores[y_bins == i]
                shuffle_splits.append(cv.split(y_scores_bin))

            # For each split iterate trough bins to collect samples
            for _ in range(n_splits):
                train_idx = []
                test_idx = []

                for i in range(len(bins)):
                    y_scores_bin = y_scores[y_bins == i]
                    n_samples_bin = len(y_scores_bin)
                    if n_samples_bin - np.ceil(test_size * n_samples_bin) <= 0:
                        continue  # skip bins with not enough points
                    indices_bin = indices[y_bins == i]
                    shuffle_split = shuffle_splits[i]
                    train_idx_bin, test_idx_bin = next(shuffle_split)
                    train_idx.extend(indices_bin[train_idx_bin])
                    test_idx.extend(indices_bin[test_idx_bin])

                train_idx = np.array(train_idx)
                test_idx = np.array(test_idx)

                yield train_idx, test_idx

        split = mysplit(y_scores, n_splits=n_splits, test_size=test_size)

    return split


def calibrate_scores(
    y_scores, y_labels, test_size=0.5, method="isotonic", max_calibration=False
):
    """Calibrate the output of a classifier.

    Fit calibrator on training set and predict on training+test set.

    Parameters
    ----------
    y_scores : (n_samples, n_classes) or (n_samples) array

    y_labels: (n_samples,) array

    test_size : float or array like or None
        Proportion of samples to use as test set. Or indices of test set.

    method : str
        Available: 'isotonic', 'sigmoid'.

    max_calibration : bool
        Whether to calibrate only the maximum scores. If True, output scores
        will be of shape (n_samples, 2).

    Returns
    -------
    y_scores_cal : (n_samples, n_classes) arrray
        Calibrated scores (containing both training and test samples).
    test_idx : array
        Indices of test samples.
    """
    y_scores = np.array(y_scores)
    y_labels = np.array(y_labels)

    n_samples = y_scores.shape[0]

    if test_size is None:
        # Test on the training set
        train_idx = np.ones(n_samples, dtype=bool)
        test_idx = np.ones(n_samples, dtype=bool)

    elif hasattr(test_size, "__len__") and not isinstance(test_size, str):
        # array like given: create train/test split from it
        test_size = np.array(test_size)
        if test_size.size != 0 and not np.can_cast(test_size, int, casting="safe"):
            raise ValueError(
                f"Values of test_size should be safely castable " "to int."
            )
        test_size = test_size.astype(int)

        if np.any(test_size >= n_samples) or np.any(test_size < 0):
            raise ValueError(
                f"test_size is an array with values out of range "
                f"[0, {n_samples-1}]."
            )

        test_idx = np.zeros(n_samples, dtype=bool)
        test_idx[test_size] = True
        train_idx = np.logical_not(test_idx)
        assert np.all(np.logical_or(train_idx, test_idx))
        assert np.sum(test_idx) == len(test_size)

    else:
        # scalar given
        cv = ShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        split = cv.split(np.zeros(n_samples))
        train_idx, test_idx = next(split)

    if max_calibration:
        # Calibrate only the maximum confidence score (weakest def of calibration)
        y_scores, y_labels = binarize_multiclass_max(y_scores, y_labels)
        y_scores = np.stack([1 - y_scores, y_scores], axis=1)

    y_scores_train = y_scores[train_idx]
    y_labels_train = y_labels[train_idx]

    class DummyClassifier(BaseEstimator):
        def __init__(self):
            self.classes_ = np.unique(y_labels)

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return X

    estimator = DummyClassifier()
    calibrated_clf = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    calibrated_clf.fit(y_scores_train, y_labels_train)
    y_scores_cal = calibrated_clf.predict_proba(y_scores)

    eps = 1e-16
    y_scores_cal = np.clip(y_scores_cal, eps, 1 - eps)

    test_idx = np.where(test_idx)[0]
    return y_scores_cal, test_idx


def compute_classif_metrics(y_scores, y_labels, y_true_probas=None):
    y_scores = torch.from_numpy(np.array(y_scores))
    y_labels = torch.from_numpy(np.array(y_labels))

    metrics = {
        "acc": Accuracy()(y_scores, y_labels).item(),
        "auroc": AUROC(compute_on_step=True).forward(y_scores, y_labels).item(),
        "ece": CalibrationError(norm="l1", compute_on_step=True)
        .forward(y_scores, y_labels)
        .item(),
        "mce": CalibrationError(norm="max", compute_on_step=True)
        .forward(y_scores, y_labels)
        .item(),
        "rmsce": CalibrationError(norm="l2", compute_on_step=True)
        .forward(y_scores, y_labels)
        .item(),
        "brier": brier_score_loss(y_labels, y_scores),
        "nll": log_loss(y_labels, y_scores),
    }
    metrics["msce"] = np.square(metrics["rmsce"])

    if y_true_probas is not None:
        y_true_probas = torch.from_numpy(np.array(y_true_probas))

        metrics.update(
            {
                "mse": MeanSquaredError(compute_on_step=True)
                .forward(y_scores, y_true_probas)
                .item(),
                "acc_bayes": Accuracy()(y_true_probas, y_labels).item(),
                "brier_bayes": brier_score_loss(y_labels, y_true_probas),
            }
        )

    return metrics


def save_path(dirpath, ext, order=[], **kwargs):
    os.makedirs(dirpath, exist_ok=True)
    keys = sorted(list(kwargs.keys()))
    if not set(order).issubset(keys):
        raise ValueError(f"Given order {order} should be a subset of {keys}.")

    for key in order:
        keys.remove(key)

    keys = order + keys

    def replace(x):
        if x is True:
            return "T"
        if x is False:
            return "F"
        if x is None:
            return "N"
        return x

    filename = ":".join(f"{k}={replace(kwargs[k])}" for k in keys)
    if not filename:
        filename = "fig"

    filename = filename.replace("(", ":")
    filename = filename.replace(")", "")
    filename = filename.replace(" ", "_")
    filename = filename.replace(",", ":")
    filename = filename.replace("@", "_")
    filename = filename.replace(".", "_")

    filename += f".{ext}"
    filepath = os.path.join(dirpath, filename)

    return filepath


def save_fig(fig, dirpath, ext="pdf", order=[], pad_inches=0.1, **kwargs):
    filepath = save_path(dirpath, ext=ext, order=order, **kwargs)
    fig.savefig(filepath, bbox_inches="tight", transparent=True, pad_inches=pad_inches)
    return filepath


def list_list_to_array(L, fill_value=None, dtype=None):
    """Convert a list of list of varying size into a numpy array with
    smaller shape possible.

    Parameters
    ----------
    L : list of lists.

    fill_value : any
        Value to fill the blank with.

    Returns
    -------
    a : array

    """
    max_length = max(map(len, L))
    L = [Li + [fill_value] * (max_length - len(Li)) for Li in L]
    return np.array(L, dtype=dtype)


def pad_array(a, shape, fill_value=0):
    """Pad a numpy array to the desired shape by appending values to axes.

    Parameters
    ----------
    a : array

    shape : tuple
        Desired shape. If one array has a smaller shape, an error is raised.

    Returns
    -------
    b : array
        Padded array with shape shape.
    """
    a_shape = np.array(a.shape, dtype=int)
    b_shape = np.array(shape, dtype=int)

    if len(a_shape) != len(b_shape):
        raise ValueError(
            f"Desired shape and array shape must have same "
            f"dimension. Array is {len(a_shape)}D, desired shape "
            f"is {len(b_shape)}D."
        )

    if (b_shape < a_shape).any():
        raise ValueError(
            f"Desired shape must have all its dimension at least "
            f"as large as input array. Asked shape {b_shape} on "
            f"array of shape {a_shape}."
        )

    pad_width = tuple((0, c) for c in b_shape - a_shape)
    return np.pad(a, pad_width, mode="constant", constant_values=fill_value)


def pad_arrays(L, shape=None, fill_value=0):
    """ "Pad a list of array to the desired shape by appending values to axes.

    Parameters
    ----------
    L : list of arrays.

    shape : tuple
        Desired shape. If one array has a smaller shape, an error is raised.

    fill_value : any
        Value to fill the blank with.

    """

    if shape is None:
        # Find the largest shape
        shapes = [np.array(a.shape) for a in L]
        shape = np.maximum.reduce(shapes)

    return [pad_array(a, shape, fill_value) for a in L]


def pairwise_call(L, f, symmetric=True, n_jobs=1, verbose=0):
    """Compute pairwise call of f on object of list L

    Parameters
    ----------
    L : list
        List of objects of shape n.

    f : callable (obj1, obj2) returning a float

    symmetric : bool
        Whether f(x, y) = f(y, x). If so, avoid half the computation.

    n_jobs : int

    Returns
    -------
    D : (n, n) array

    """
    n = len(L)
    D = np.full((n, n), np.nan)

    if symmetric:
        indexes = [(i, j) for i in range(n) for j in range(i, n)]
    else:
        indexes = [(i, j) for i in range(n) for j in range(n)]

    disable = verbose <= 0
    res = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(f)(L[i], L[j]) for i, j in tqdm(indexes, disable=disable)
    )

    for k, (i, j) in enumerate(tqdm(indexes, disable=disable)):
        D[i, j] = res[k]
        if symmetric:
            D[j, i] = res[k]

    return D


def _get_out_kwargs(
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
    extra_out_kwargs=dict(),
    order=None,
):
    "Helper function to build filename from arguments."

    out_kwargs = {
        "clustering": clustering,
        "n_bins": n_bins,
        "ci": ci,
        "hist": hist,
        "test_size": test_size,
        "calibrate": calibrate,
    }

    _order = ["clustering"]

    if name is not None:
        out_kwargs["name"] = name
        _order.insert(0, "name")

    if not isinstance(clustering, str):
        out_kwargs["clustering"] = "manual"

    elif clustering == "decision_tree":
        out_kwargs["min_samples_leaf"] = min_samples_leaf
        out_kwargs["max_clusters_bin"] = max_clusters_bin
        _order.append("max_clusters_bin")
        _order.append("test_size")
        _order.append("min_samples_leaf")

    elif clustering == "kmeans":
        out_kwargs["n_clusters"] = n_clusters
        out_kwargs["min_cluster_size"] = min_cluster_size
        _order.append("n_clusters")

    out_kwargs.update(extra_out_kwargs)

    if order is not None:
        _order = order

    return out_kwargs, _order


class ProgressBar:
    """Taken from https://stackoverflow.com/a/53643011"""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def get_md5(filepath):
    """Taken from https://stackoverflow.com/a/3431838/18429836."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
