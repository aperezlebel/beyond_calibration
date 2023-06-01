"""Define base class for loading and forwarding networks."""
import os
import sys
from abc import ABC, abstractmethod
from itertools import islice
from multiprocessing import Value
from os.path import basename, exists, join, splitext

import torch
from joblib import Memory, Parallel, delayed
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

memory = Memory("joblib_cache")


class CustomDatasetFolder(datasets.DatasetFolder):
    @staticmethod
    def make_dataset(directory, class_to_idx, extensions, is_valid_file):
        # Retrieve original class idx in the filename of the sample
        class_to_idx = {k: eval(k.split("_")[1]) for k, _ in class_to_idx.items()}
        return datasets.DatasetFolder.make_dataset(
            directory, class_to_idx, extensions, is_valid_file
        )


class BaseNet(ABC):
    def __init__(self):
        self.model = self.create_model()
        self.model.eval()
        self.truncated_model, self.last_layer = self.create_truncated_model()
        self.truncated_model.eval()
        if self.last_layer is not None:
            self.last_layer.eval()
        self._get_Xt_y_scores_labels = memory.cache(
            self._get_Xt_y_scores_labels, ignore=["self"]
        )

    @abstractmethod
    def create_model():
        pass

    @abstractmethod
    def create_truncated_model():
        pass

    def get_default_dirpath(self):
        dirpath = join("datasets/", str(self))
        return dirpath

    def forward_truncated(self, input):
        return self.truncated_model(input)

    def forward_whole(self, input):
        return self.model(input)

    def forward(self, input, truncated=False):
        if truncated:
            return self.forward_truncated(input)
        return self.forward_whole(input)

    def forward_dataset(
        self,
        dirpath=None,
        batch_size=1,
        n_batch_max=None,
        verbose=0,
        n_jobs=1,
        skip_exist=False,
        position=0,
    ):
        """Forward dataset until the last layer of the network.

        Parameters
        ----------
        dirpath : str
            Path to directory in which to store the forwarded dataset.

        batch_size : int
            Batch size of the loader reading the dataset.

        n_batch_max : int or None
            Prematurely stop loading the dataset.

        verbose : int
            Verbosity level.

        n_jobs : int
            Number of jobs to run in parallel.

        skip_exist : bool
            Skip if destination directory exists.

        position : int
            Progress bar offset (useful for multithreading).

        """
        loader = self.get_loader(batch_size=batch_size, shuffle=False)
        class_names = self.get_class_names()

        ds = loader.dataset
        samples = ds.samples if hasattr(ds, "samples") else None

        if dirpath is None:
            dirpath = self.get_default_dirpath()

        if skip_exist and os.path.exists(dirpath):
            n_samples = len(ds)
            try:
                n_samples_target = len(
                    CustomDatasetFolder(dirpath, loader=torch.load, extensions=".pt")
                )
            except FileNotFoundError:
                n_samples_target = 0

            if n_samples == n_samples_target:
                print(
                    f"forward_dataset skipped because {dirpath} exists with "
                    f"size {n_samples} (>= {n_samples} in source)."
                )
                return

            if n_samples_target > n_samples:
                raise ValueError(
                    f"Target dataset exists at {dirpath} "
                    f"but has more samples than original one. "
                    f"({n_samples_target} > {n_samples})."
                )

            print(
                f"forward_dataset not skipped because target {dirpath} "
                f"has {n_samples_target} samples and source has "
                f"{n_samples}."
            )

        def forward_one(i, it):
            input, y_labels = self.loader_iter_to_input_label(it)
            Xt = self.forward(input, truncated=True).detach()
            Xt = torch.atleast_2d(Xt)

            if self.last_layer is None:
                # Some model does not give access to their last layer
                # and can only be evaluated. So need to compute
                # logits at first forward.
                y_logits = self.forward(input, truncated=False).detach()
                y_logits = torch.atleast_2d(y_logits)
            else:
                y_logits = None

            for j in range(y_labels.shape[0]):
                k = i * y_labels.shape[0] + j
                label = y_labels[j].item()
                class_dirname = f"{class_names[label]}_{label}"
                if samples is not None:
                    filename = splitext(basename(samples[k][0]))[0]
                else:
                    filename = f"{class_names[label]}_{k}"

                subdirpath = join(dirpath, class_dirname)
                os.makedirs(subdirpath, exist_ok=True)
                if self.last_layer is None:
                    out = (Xt[j, :], y_logits[j, :])
                else:
                    out = Xt[j, :]
                torch.save(out, join(subdirpath, f"{filename}.pt"))

        n_iter = len(loader) if n_batch_max is None else min(len(loader), n_batch_max)
        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(forward_one)(i, it)
            for i, it in tqdm(
                islice(enumerate(loader), n_batch_max),
                total=n_iter,
                disable=(verbose <= 0),
                position=position,
                desc=str(self),
            )
        )

    def load_forwarded_dataset(
        self,
        dirpath=None,
        batch_size=1,
        n_batch_max=None,
        dump=True,
        verbose=0,
        n_jobs=1,
        skip_exist=False,
        position=0,
    ):
        """Load a previously forwarded dataset.

        Parameters
        ----------
        dirpath : str
            Path to directory from which to read the forwarded dataset.

        batch_size : int
            Batch size of the loader reading the dataset.

        n_batch_max : int or None
            Prematurely stop loading the dataset.

        dump : bool
            Whether to dump the aggregated tensors.

        verbose : int
            Verbosity level.

        n_jobs : int
            Number of jobs to run in parallel.

        skip_exist : bool
            Skip if destination directory exists.

        position : int
            Progress bar offset (useful for multithreading).

        """
        if dirpath is None:
            dirpath = self.get_default_dirpath()

        if skip_exist and (
            exists(join(dirpath, "Xt.pt"))
            and exists(join(dirpath, "y_scores.pt"))
            and exists(join(dirpath, "y_labels.pt"))
        ):
            print(
                f"load_forwarded_dataset skipped because Xt.pt, "
                f"y_scores.pt and y_labels.pt exist in {dirpath}."
            )
            return

        ds = CustomDatasetFolder(dirpath, loader=torch.load, extensions=".pt")
        loader = DataLoader(ds, batch_size=batch_size)

        # Store cumulative size of loaded tensors
        cumulative_size = Value("i", 0)

        def load_one(Xt, y_labels, cumulative_size, pbar):
            if self.last_layer is not None:
                y_logits = self.last_layer(Xt)
            else:
                # y_logits has been computed and stored during forward
                Xt, y_logits = Xt

            y_scores = self.logits_to_scores(y_logits)

            Xt = Xt.detach()
            y_scores = y_scores.detach()
            y_labels = y_labels.detach()

            # Compute size of tensors stored so far for the progress bar
            with cumulative_size.get_lock():
                cumulative_size.value += (
                    sys.getsizeof(Xt.storage())
                    + sys.getsizeof(y_scores.storage())
                    + sys.getsizeof(y_labels.storage())
                )
                pbar.set_description(f"{int(cumulative_size.value/1e6)}MB")

            return Xt, y_scores, y_labels

        n_iter = len(loader) if n_batch_max is None else min(len(loader), n_batch_max)
        pbar = tqdm(
            islice(loader, n_batch_max),
            total=n_iter,
            disable=(verbose <= 0),
            position=position,
            desc=str(self),
        )
        res = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(load_one)(Xt, y_labels, cumulative_size, pbar)
            for Xt, y_labels in pbar
        )

        # Zip results from list of tuples to tuple of lists
        L_Xt, L_y_scores, L_y_labels = list(zip(*res))

        Xt = torch.cat(L_Xt, axis=0)
        y_scores = torch.cat(L_y_scores, axis=0)
        y_labels = torch.cat(L_y_labels, axis=0)

        if dump:
            torch.save(Xt, join(dirpath, "Xt.pt"))
            torch.save(y_scores, join(dirpath, "y_scores.pt"))
            torch.save(y_labels, join(dirpath, "y_labels.pt"))

        return Xt, y_scores, y_labels

    def _get_Xt_y_scores_labels(self, batch_size=1, n_batch_max=None, name=None):
        loader = self.get_loader(batch_size=batch_size)

        L_Xt = []
        L_y_scores = []
        L_y_labels = []

        for i, it in tqdm(enumerate(loader), total=len(loader)):
            if n_batch_max is not None and i >= n_batch_max:
                break

            input, y_labels = self.loader_iter_to_input_label(it)
            Xt = self.forward(input, truncated=True)
            y_logits = self.last_layer(Xt)
            y_scores = self.logits_to_scores(y_logits)
            L_Xt.append(Xt.detach())
            L_y_scores.append(y_scores.detach())
            L_y_labels.append(y_labels.detach())

        Xt = torch.cat(L_Xt, axis=0)
        y_scores = torch.cat(L_y_scores, axis=0)
        y_labels = torch.cat(L_y_labels, axis=0)

        return Xt, y_scores, y_labels

    def get_Xt_y_scores_labels(self, batch_size=1, n_batch_max=None):
        return self._get_Xt_y_scores_labels(
            batch_size=batch_size,
            n_batch_max=n_batch_max,
            name=self.__class__.__name__,
        )

    @abstractmethod
    def get_w(self):
        pass

    @abstractmethod
    def get_intercept(self):
        pass

    @abstractmethod
    def get_class_names(self):
        pass

    @abstractmethod
    def logits_to_scores(self):
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass

    def get_loader(self, batch_size=1, shuffle=False, seed=None):
        generator = torch.Generator() if shuffle else None
        if shuffle and seed is not None:
            generator.manual_seed(seed)

        try:
            ds = self.get_dataset()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}\nMake sure you downloaded and extracted the datasets."
            )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, generator=generator
        )

    def loader_iter_to_input_label(self, iter):
        return iter

    def get_class_name(self, with_type=False):
        type = f":{self.type}" if with_type and hasattr(self, "type") else ""
        return f"{self.__class__.__name__.lower()}{type}"

    def __repr__(self):
        return f"{self.get_dataset_name()}@{self.get_class_name(True)}"
