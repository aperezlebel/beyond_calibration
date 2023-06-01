import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from ._utils import list_list_to_array, pad_arrays


def cluster_evaluate(
    X,
    y_labels,
    y_scores,
    bins=15,
    clustering="kmeans",
    n_clusters=2,
    min_samples_leaf=None,
    max_clusters_bin=2,
    n_samples_per_cluster_per_bin=None,
    test_size=0.5,
    return_clustering=None,
    verbose=0,
    n_jobs=1,
):
    """Evaluate fraction of positives in clustered bins.

    Parameters
    ----------
    X : (n, d) array
        The data samples.

    y_labels : (n,) array
        The data labels. Must be binary.

    y_scores : (n,) array
        The scores given to the positive class.

    bins : int or array
        Number of bins or bins.

    clustering : str or (n,) array
        Clustering method to use. Choices: 'kmeans', 'decision_tree' or
        a size (n,) array of cluster assignations (ie all samples with the same
        value belong to the same cluster).

    n_clusters : int
        Number of clusters in each bin. Used for clustering='kmeans' only.

    min_samples_leaf : int
        Parameters passed to DecisionTreeRegressor when
        clustering='decision_tree'. Ignored if max_clusters_bin is not None.

    max_clusters_bin : int
        Compute min_samples_leaf per leaf when clustering='decision_tree'.

    test_size : float or array or None
        Whether to train/test split data for the clustering. If float given,
        the size of the test set as a propotion. If None: no train/test split.
        If array: the sample indicices to take as test set. Train set is
        the remaining.

    return_clustering : str or None
        Whether to return cluster assignments. None: don't return.
        'macro': cluster labels are common to all bins.
        'micro': cluster labels are unique.

    verbose : int
        Verbosity level.

    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    frac_pos : (bins, C) array
        The fraction of positives in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    counts : (bins, C) array
        The number of samples in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    mean_scores : (bins, C) array
        The mean score of samples in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    """
    X = np.array(X)
    y_labels = np.array(y_labels)
    y_scores = np.array(y_scores)

    if X.shape[0] != y_labels.shape[0]:
        raise ValueError(
            f"Shape mismatch between X {X.shape} and y_labels {y_labels.shape}"
        )

    if X.shape[0] != y_scores.shape[0]:
        raise ValueError(
            f"Shape mismatch between X {X.shape} and y_scores {y_scores.shape}"
        )

    unique_labels = np.unique(y_labels)
    if not np.isin(unique_labels, [0, 1]).all():
        raise ValueError(
            f"y_labels must take values in {{0, 1}}. Found {unique_labels}."
        )

    valid_clustering_strategy = ["kmeans", "decision_tree", None]
    if clustering is not None and (
        not isinstance(clustering, str) or clustering not in valid_clustering_strategy
    ):
        try:
            _clustering = np.array(clustering)
            _clustering_shape = _clustering.shape
            clustering = "manual"
        except AttributeError:
            raise ValueError(
                f"{clustering} is an invalid clustering strategy."
                f" Choices are: {valid_clustering_strategy} or "
                f"array of shape {y_labels.shape}"
            )

        if _clustering_shape != y_labels.shape:
            raise ValueError(
                f"If array given, clustering must be of same"
                f"shape as y_labels {y_labels.shape}. Given "
                f"{_clustering.shape}."
            )

    if (
        clustering == "decision_tree"
        and max_clusters_bin is not None
        and min_samples_leaf is not None
    ):
        raise ValueError(
            f"max_clusters_bin and min_samples_leaf cannot "
            f"be both not None. Got {max_clusters_bin} "
            f"and {min_samples_leaf}."
        )

    if (
        clustering == "decision_tree"
        and max_clusters_bin is None
        and min_samples_leaf is None
        and n_samples_per_cluster_per_bin is None
    ):
        raise ValueError(
            f"max_clusters_bin and min_samples_leaf and "
            f"n_samples_per_cluster_per_bin cannot "
            f"be all None."
        )

    if hasattr(test_size, "__len__") and not isinstance(test_size, str):
        # array like given: create train/test split from it
        test_size = np.array(test_size)
        if test_size.size != 0 and not np.can_cast(test_size, int, casting="safe"):
            raise ValueError(
                f"Values of test_size should be safely castable " "to int."
            )
        test_size = test_size.astype(int)

        if np.any(test_size >= X.shape[0]) or np.any(test_size < 0):
            raise ValueError(
                f"test_size is an array with values out of range "
                f"[0, {X.shape[0]-1}]."
            )

        test_idx = np.zeros(X.shape[0], dtype=bool)
        test_idx[test_size] = True
        train_idx = np.logical_not(test_idx)
        assert np.all(np.logical_or(train_idx, test_idx))
        assert np.sum(test_idx) == len(test_size)

    else:
        train_idx = None
        test_idx = None

    try:
        bins = np.linspace(0, 1, bins + 1)
    except TypeError:
        pass

    n_bins = len(bins) - 1

    y_bins = np.digitize(y_scores, bins=bins) - 1
    y_bins = np.clip(y_bins, a_min=None, a_max=n_bins - 1)

    frac_pos = [[] for _ in range(n_bins)]
    mean_scores = [[] for _ in range(n_bins)]
    counts = [[] for _ in range(n_bins)]

    cluster_assignments_train = np.full_like(y_labels, np.nan, dtype=float)
    cluster_assignments_test = np.full_like(y_labels, np.nan, dtype=float)
    idx = np.arange(len(y_labels), dtype=int)

    def cluster_one(i, max_clusters_bin=max_clusters_bin, n_clusters=n_clusters):
        frac_pos = []
        mean_scores = []
        counts = []
        frac_pos_train = []
        mean_scores_train = []
        counts_train = []

        X_bin = X[y_bins == i, :]
        y_labels_bin = y_labels[y_bins == i]
        y_scores_bin = y_scores[y_bins == i]

        n_samples = len(y_labels_bin)

        if test_size is None or n_samples == 0:
            train_idx_bin = np.ones_like(y_labels_bin, dtype=bool)
            test_idx_bin = np.ones_like(y_labels_bin, dtype=bool)

        elif train_idx is not None or test_idx is not None:
            train_idx_bin = train_idx[y_bins == i]
            test_idx_bin = test_idx[y_bins == i]

        else:
            if n_samples - np.ceil(test_size * n_samples) > 0:
                shuffle_split = ShuffleSplit(
                    n_splits=1, test_size=test_size, random_state=0
                )
                train_idx_bin, test_idx_bin = next(shuffle_split.split(y_labels_bin))

            else:
                train_idx_bin = np.ones_like(y_labels_bin, dtype=bool)
                train_idx_bin[-1] = 0
                test_idx_bin = np.zeros_like(y_labels_bin, dtype=bool)
                test_idx_bin[-1] = 1

        X_bin_train = X_bin[train_idx_bin, :]
        X_bin_test = X_bin[test_idx_bin, :]
        y_labels_bin_train = y_labels_bin[train_idx_bin]
        y_labels_bin_test = y_labels_bin[test_idx_bin]
        y_scores_bin_train = y_scores_bin[train_idx_bin]
        y_scores_bin_test = y_scores_bin[test_idx_bin]
        labels_train = []
        labels_test = []

        if verbose > 1:
            print(
                f"Bin {i+1}/{n_bins} ({bins[i]:.2f} - {bins[i+1]:.2f}):\t{X_bin.shape[0]} samples"
            )

        if n_samples_per_cluster_per_bin is not None:
            # Deduce n_clusters from n_samples_per_cluster_per_bin
            n_samples_bin_test = X_bin_test.shape[0]
            n_clusters = n_samples_bin_test // n_samples_per_cluster_per_bin
            n_clusters = int(max(n_clusters, 1))
            max_clusters_bin = n_clusters

        # Convert max_clusters_bin into min_samples_leaf
        # (only used for decision_tree and manual clustering with train split)
        if max_clusters_bin is not None:
            _min_samples_leaf = X_bin_train.shape[0] // max_clusters_bin
            _min_samples_leaf = int(max(_min_samples_leaf, 1))
        else:
            _min_samples_leaf = int(min_samples_leaf)

        if len(X_bin_train) == 0:
            # cant learn clusters: put every test sample in same cluster and
            # skip clustering
            labels_train = []
            labels_test = [0] * len(X_bin_test)

        elif clustering == "kmeans" and X_bin_train.shape[0] <= n_clusters:
            # cant learn clusters, put every training sample in one cluster
            # and do the same on test samples
            labels_train = [0] * len(X_bin_train)
            labels_test = [0] * len(X_bin_test)

        elif clustering == "kmeans":
            estimator = KMeans(n_clusters=n_clusters, random_state=0)

            # Cluster assignment
            labels_train = estimator.fit_predict(X_bin_train)
            if len(X_bin_test) != 0:
                labels_test = estimator.predict(X_bin_test)

        elif clustering == "decision_tree":
            estimator = DecisionTreeRegressor(
                min_samples_leaf=_min_samples_leaf, random_state=0
            )
            estimator.fit(X_bin_train, y_labels_bin_train)

            # Cluster assignment
            labels_train = estimator.apply(X_bin_train)
            if len(X_bin_test) != 0:
                labels_test = estimator.apply(X_bin_test)

        elif clustering == "manual" and (
            test_size is None or isinstance(test_size, np.ndarray)
        ):
            _clustering_bin = _clustering[y_bins == i]
            labels_train = _clustering_bin[train_idx_bin]
            labels_test = _clustering_bin[test_idx_bin]

        elif clustering == "manual" and test_size is not None:
            _clustering_bin = _clustering[y_bins == i]
            labels_train = _clustering_bin[train_idx_bin]

            # Maps given labels to integers from 0 to n_unique_labels - 1
            unique_labels = np.unique(_clustering)
            for k, _label in enumerate(unique_labels):
                labels_train[labels_train == _label] = k

            estimator = HistGradientBoostingClassifier(
                min_samples_leaf=_min_samples_leaf, random_state=0
            )
            estimator.fit(X_bin_train, labels_train)

            if len(X_bin_test) != 0:
                labels_test = estimator.predict(X_bin_test)

        elif clustering is None:
            labels_train = np.zeros_like(y_labels_bin_train, dtype=int)
            labels_test = np.zeros_like(y_labels_bin_test, dtype=int)

        # From cluster assignments, compute frac_pos, mean_scores and counts
        unique_labels_test, unique_counts_test = np.unique(
            labels_test, return_counts=True
        )
        unique_labels_train, unique_counts_train = np.unique(
            labels_train, return_counts=True
        )

        for k, label in enumerate(unique_labels_test):
            if len(labels_test == label) > 0:
                frac_pos.append(np.mean(y_labels_bin_test[labels_test == label]))
                mean_scores.append(np.mean(y_scores_bin_test[labels_test == label]))

        for k, label in enumerate(unique_labels_test[np.argsort(np.array(frac_pos))]):
            if len(labels_test == label) > 0:
                cluster_assignments_test[
                    idx[y_bins == i][test_idx_bin][labels_test == label]
                ] = k

        for k, label in enumerate(unique_labels_train):
            if len(labels_train == label) > 0:
                frac_pos_train.append(
                    np.mean(y_labels_bin_train[labels_train == label])
                )
                mean_scores_train.append(
                    np.mean(y_scores_bin_train[labels_train == label])
                )

        for k, label in enumerate(
            unique_labels_train[np.argsort(np.array(frac_pos_train))]
        ):
            if len(labels_test == label) > 0:
                cluster_assignments_train[
                    idx[y_bins == i][train_idx_bin][labels_train == label]
                ] = k

        counts.extend(unique_counts_test)
        counts_train.extend(unique_counts_train)

        return (
            frac_pos,
            counts,
            mean_scores,
            frac_pos_train,
            counts_train,
            mean_scores_train,
        )

    res = Parallel(n_jobs=n_jobs)(
        delayed(cluster_one)(i) for i in tqdm(range(n_bins), disable=(verbose != 1))
    )

    # Zip results from list of tuples to tuple of lists
    (
        frac_pos,
        counts,
        mean_scores,
        frac_pos_train,
        counts_train,
        mean_scores_train,
    ) = list(zip(*res))

    frac_pos = list_list_to_array(frac_pos, fill_value=0)
    mean_scores = list_list_to_array(mean_scores, fill_value=0)
    counts = list_list_to_array(counts, fill_value=0, dtype=int)
    frac_pos_train = list_list_to_array(frac_pos_train, fill_value=0)
    mean_scores_train = list_list_to_array(mean_scores_train, fill_value=0)
    counts_train = list_list_to_array(counts_train, fill_value=0, dtype=int)

    if test_size is None:
        if return_clustering is not None:
            return frac_pos, counts, mean_scores, cluster_assignments_test
        return frac_pos, counts, mean_scores

    if return_clustering is not None:
        return (
            frac_pos,
            counts,
            mean_scores,
            cluster_assignments_test,
            frac_pos_train,
            counts_train,
            mean_scores_train,
            cluster_assignments_train,
        )
    return (
        frac_pos,
        counts,
        mean_scores,
        frac_pos_train,
        counts_train,
        mean_scores_train,
    )


def cluster_evaluate_marginals(
    Xt,
    y_labels,
    y_scores,
    positive_class,
    bins=10,
    n_clusters=2,
    clustering="kmeans",
    verbose=0,
    min_samples_leaf=1,
    max_clusters_bin=None,
    n_samples_per_cluster_per_bin=None,
    test_size=None,
    n_jobs=1,
):
    """Evaluate fraction of positives in clustered bins.

    Parameters
    ----------
    X : (n, d) array
        The data samples.

    y_labels : (n,) array
        The data labels. Must be integers in {0, ..., K}.

    y_scores : (n, K) array
        The scores given to each of the K classes.

    positive_class : int or None
        The one of the K classes to consider as the positive class.
        If None, the output arrays are 3D with K as last dimension.

    bins : int or array
        Number of bins or bins.

    n_clusters : int
        Number of clusters in each bin.

    min_samples_leaf : int
        Parameters passed to DecisionTreeRegressor when
        clustering='decision_tree'. Ignored if max_clusters_bin is not None.

    max_clusters_bin : int
        Compute min_samples_leaf per leaf when clustering='decision_tree'.

    verbose : int
        Verbosity level.

    n_jobs : int
        Number of jobs to run in parallel. Only used when positive_class
        is None.

    test_size : float or None
        Whether to train/test split data for the clustering. If float given,
        the size of the test set as a propotion. If None: no train/test split.

    Returns
    -------
    frac_pos : (bins, C) or (bins, n_clusters, K) array
        The fraction of positives in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    counts : (bins, C) or (bins, n_clusters, K) array
        The number of samples in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    mean_scores : (bins, C) or (bins, n_clusters, K) array
        The mean score of samples in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    """
    Xt = np.array(Xt)
    y_labels = np.array(y_labels)
    y_scores = np.array(y_scores)

    if y_scores.shape[1] <= 1:
        raise ValueError(
            f"y_scores must have at least 2 classes. " f"Found shape {y_scores.shape}."
        )

    n_classes = y_scores.shape[1]
    unique_labels = np.unique(y_labels)
    if not np.isin(unique_labels, range(n_classes)).all():
        raise ValueError(
            f"y_labels must take values in {{0, ..., {n_classes-1}}}. Found {unique_labels}."
        )

    def evaluate_one(positive_class):
        y_labels_k = (y_labels == positive_class).astype(int)
        y_scores_k = y_scores[:, positive_class]

        (frac_pos, counts, mean_scores, *_) = cluster_evaluate(
            Xt,
            y_labels_k,
            y_scores_k,
            bins=bins,
            n_clusters=n_clusters,
            clustering=clustering,
            min_samples_leaf=min_samples_leaf,
            max_clusters_bin=max_clusters_bin,
            n_samples_per_cluster_per_bin=n_samples_per_cluster_per_bin,
            verbose=verbose - 1,
            n_jobs=1,
            test_size=test_size,
        )
        return frac_pos, counts, mean_scores

    if positive_class is not None:
        if positive_class not in range(n_classes):
            raise ValueError(
                f"positive_class must be in {{0, ..., {n_classes-1}}}. Given {positive_class}."
            )

        return evaluate_one(positive_class)

    # If positive_class is None, compute for all classes
    res = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_one)(positive_class)
        for positive_class in tqdm(range(n_classes), disable=(verbose <= 0))
    )

    # Zip results from list of tuples to tuple of lists
    L_frac_pos, L_counts, L_mean_scores = list(zip(*res))

    L_frac_pos = pad_arrays(L_frac_pos)
    L_counts = pad_arrays(L_counts)
    L_mean_scores = pad_arrays(L_mean_scores)

    frac_pos = np.stack(L_frac_pos, axis=2)
    counts = np.stack(L_counts, axis=2)
    mean_scores = np.stack(L_mean_scores, axis=2)

    return frac_pos, counts, mean_scores


def cluster_evaluate_max(
    Xt,
    y_labels,
    y_scores,
    bins=10,
    n_clusters=2,
    breakout=False,
    verbose=0,
    clustering="kmeans",
    min_samples_leaf=1,
    max_clusters_bin=None,
    n_samples_per_cluster_per_bin=None,
    test_size=None,
    n_jobs=1,
    binary_scores=False,
    return_clustering=False,
):
    """Evaluate fraction of positives in clustered bins.

    Parameters
    ----------
    X : (n, d) array
        The data samples.

    y_labels : (n,) array
        The data labels. Must be integers in {0, ..., K}.

    y_scores : (n, K) array
        The scores given to each of the K classes.

    bins : int or array
        Number of bins or bins.

    n_clusters : int
        Number of clusters in each bin.

    min_samples_leaf : int
        Parameters passed to DecisionTreeRegressor when
        clustering='decision_tree'. Ignored if max_clusters_bin is not None.

    max_clusters_bin : int
        Compute min_samples_leaf per leaf when clustering='decision_tree'.

    breakout : bool
        Whether to breakout per class after binning to run the clustering.

    verbose : int
        Verbosity level.

    test_size : float or None
        Whether to train/test split data for the clustering. If float given,
        the size of the test set as a propotion. If None: no train/test split.

    n_jobs : int
        Number of jobs to run in parallel. Only used when breakout=True.

    binary_scores : bool
        Whether the given y_scores and y_labels are already binarized in case
        of a multiclass problem. Ie y_scores is the maximum score
        and y_labels are 1 if the predicted class is right and 0 if it is wrong.

    return_clustering : bool
        Whether to return cluster assignments.

    Returns
    -------
    frac_pos : (bins, C, K) array
        The fraction of positives of a class in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    counts : (bins, C, K) array
        The number of samples of a class in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    mean_scores : (bins, C, K) array
        The mean score of samples of a class in each cluster for each bin.
        C=n_clusters if clustering='kmeans'.

    """
    if y_scores.ndim == 1:
        raise ValueError(f"y_scores must bet 2D. Got shape {y_scores.shape}.")

    if y_scores.shape[1] <= 1:
        raise ValueError(
            f"y_scores must have at least 2 classes. " f"Found shape {y_scores.shape}."
        )

    n_classes = y_scores.shape[1]
    unique_labels = np.unique(y_labels)
    if not np.isin(unique_labels, range(n_classes)).all():
        raise ValueError(
            f"y_labels must take values in {{0, ..., {n_classes-1}}}. Found {unique_labels}."
        )

    Xt = np.array(Xt)
    y_labels = np.array(y_labels)
    y_scores = np.array(y_scores)

    if binary_scores:
        y_scores_max = y_scores[:, 1]
        y_labels_binarized = y_labels
    else:
        y_scores_max = np.max(np.array(y_scores), axis=1)
        y_labels_pred = np.argmax(y_scores, axis=1)
        y_labels_binarized = (y_labels_pred == y_labels).astype(int)

    if not breakout:
        return cluster_evaluate(
            Xt,
            y_labels_binarized,
            y_scores_max,
            bins=bins,
            n_clusters=n_clusters,
            clustering=clustering,
            min_samples_leaf=min_samples_leaf,
            max_clusters_bin=max_clusters_bin,
            n_samples_per_cluster_per_bin=n_samples_per_cluster_per_bin,
            verbose=verbose,
            n_jobs=n_jobs,
            test_size=test_size,
            return_clustering=return_clustering,
        )

    if binary_scores:
        raise ValueError(f"binary_scores cant be true when breakout is true.")

    # Breakout points per class before clustering
    def evaluate_one(positive_class):
        y_labels_binarized = (y_labels == positive_class).astype(int)
        idx_pos = y_labels_pred == positive_class

        (frac_pos, counts, mean_scores, *_) = cluster_evaluate(
            Xt[idx_pos, :],
            y_labels_binarized[idx_pos],
            y_scores_max[idx_pos],
            bins=bins,
            n_clusters=n_clusters,
            clustering=clustering,
            min_samples_leaf=min_samples_leaf,
            max_clusters_bin=max_clusters_bin,
            n_samples_per_cluster_per_bin=n_samples_per_cluster_per_bin,
            verbose=verbose - 1,
            n_jobs=1,
            test_size=test_size,
        )
        return frac_pos, counts, mean_scores

    res = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_one)(positive_class)
        for positive_class in tqdm(range(n_classes), disable=(verbose <= 0))
    )

    # Zip results from list of tuples to tuple of lists
    L_frac_pos, L_counts, L_mean_scores = list(zip(*res))

    L_frac_pos = pad_arrays(L_frac_pos)
    L_counts = pad_arrays(L_counts)
    L_mean_scores = pad_arrays(L_mean_scores)

    frac_pos = np.concatenate(L_frac_pos, axis=1)
    counts = np.concatenate(L_counts, axis=1)
    mean_scores = np.concatenate(L_mean_scores, axis=1)

    return frac_pos, counts, mean_scores
