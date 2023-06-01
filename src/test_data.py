import itertools
import os
import re
from os.path import join

import numpy as np
import pytest
from joblib import Parallel, delayed
from tqdm import tqdm

from src._utils import ProgressBar, get_md5
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

best_versions = {
    IMAGENET_VGG: "19_bn",
    IMAGENET_ResNet: "152",
    IMAGENET_DenseNet: "161",
    IMAGENET_ShuffleNet: "1_0",
    IMAGENET_MobileNet: "v3L",
    IMAGENET_ResNext: "101",
    IMAGENET_WideResNet: "101",
    IMAGENET_MNASNet: "1_0",
    IMAGENET_EfficientNet: "b7",
    IMAGENET_RegNet: "y_32gf",
    IMAGENET_VisionTransformer: "l_16",
    IMAGENET_ConvNeXt: "large",
}

urls = {
    "ILSVRC2012_devkit_t12.tar.gz": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
    "ILSVRC2012_img_val.tar": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    "imagenet-r.tar": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
    "imagenet-c/blur.tar": "https://zenodo.org/record/2235448/files/blur.tar",
    "imagenet-c/digital.tar": "https://zenodo.org/record/2235448/files/digital.tar",
    "imagenet-c/extra.tar": "https://zenodo.org/record/2235448/files/extra.tar",
    "imagenet-c/noise.tar": "https://zenodo.org/record/2235448/files/noise.tar",
    "imagenet-c/weather.tar": "https://zenodo.org/record/2235448/files/weather.tar",
}

md5sums = {
    "ILSVRC2012_devkit_t12.tar.gz": "fa75699e90414af021442c21a62c3abf",
    "ILSVRC2012_img_val.tar": "29b22e2961454d5413ddabcf34fc5622",
    "imagenet-r.tar": "a61312130a589d0ca1a8fca1f2bd3337",
    "imagenet-c/blur.tar": "2d8e81fdd8e07fef67b9334fa635e45c",
    "imagenet-c/digital.tar": "89157860d7b10d5797849337ca2e5c03",
    "imagenet-c/extra.tar": "d492dfba5fc162d8ec2c3cd8ee672984",
    "imagenet-c/noise.tar": "e80562d7f6c3f8834afb1ecf27252745",
    "imagenet-c/weather.tar": "33ffea4db4d93fe4a428c40a6ce0c25d",
}

to_untar = list(urls.keys())[2:]

nets = [
    IMAGENET_VGG,
    IMAGENET_ResNet,
    IMAGENET_DenseNet,
    IMAGENET_Inception,
    IMAGENET_GoogLeNet,
    IMAGENET_ShuffleNet,
    IMAGENET_MobileNet,
    IMAGENET_ResNext,
    IMAGENET_WideResNet,
    IMAGENET_MNASNet,
    IMAGENET_EfficientNet,
    IMAGENET_RegNet,
    IMAGENET_VisionTransformer,
    IMAGENET_AlexNet,
    IMAGENET_ConvNeXt,
]

splits = [
    "test_c:_merged_no_rep5",
    "test_r",
    "test_c:snow5",
    "val",
]

bests = [
    False,
    True,
]


def check_md5sum(name, path, raise_error=True):
    print("\tChecking md5sum...", end=" ", flush=True)
    if md5sums[name] is not None:
        current_md5 = get_md5(path)
        expected_md5 = md5sums[name]
        if current_md5 != expected_md5:
            if raise_error:
                raise ValueError(
                    f'Downloaded archive "{name}" has wrong md5sum.\n'
                    f"Current: {current_md5}\n"
                    f"Expected: {expected_md5}\n"
                    f"The file may be corrupted. Delete {name} and try again.\n"
                    f"If the problem persists, the source may have changed.\n"
                    f"{urls[name]}"
                )
            return False
        print("OK.")
    else:
        print("SKIPPED: md5sum not provided.")
    return True


def test_download_datasets():
    test_download_vision_datasets()

    # Download NLP dataset
    for split in ["test_seen", "test_unseen"]:
        test_download_nlp_network(split, 0)


def test_download_vision_datasets(splits=None):
    import os
    import urllib.request

    split_to_names = {
        "val": ["ILSVRC2012_devkit_t12.tar.gz", "ILSVRC2012_img_val.tar"],
        "test_r": [
            "ILSVRC2012_devkit_t12.tar.gz",
            "ILSVRC2012_img_val.tar",
            "imagenet-r.tar",
        ],
        "test_c:snow5": [
            "ILSVRC2012_devkit_t12.tar.gz",
            "ILSVRC2012_img_val.tar",
            "imagenet-c/blur.tar",
            "imagenet-c/digital.tar",
            "imagenet-c/extra.tar",
            "imagenet-c/noise.tar",
            "imagenet-c/weather.tar",
        ],
    }
    split_to_names["test_c:_merged_no_rep5"] = split_to_names["test_c:snow5"]

    if splits is None:
        names = urls.keys()

    else:
        names = set(
            itertools.chain.from_iterable([split_to_names[split] for split in splits])
        )

    for name in names:
        url = urls[name]
        path = os.path.join("datasets/", name)
        print(f"\n{path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Download
        print(f"\tDownloading...", end=" ", flush=True)

        if os.path.exists(path) and check_md5sum(name, path, raise_error=False):
            print("\tSKIPPED: file exists with valid md5sum.")

        else:
            # File does not exist or has an invalid md5sum: download
            print()
            urllib.request.urlretrieve(url, path, ProgressBar())
            print("\tOK.")

            # Check md5sum
            check_md5sum(name, path)


@pytest.mark.parametrize(
    "name",
    [
        "imagenet-r.tar",
        "imagenet-c/blur.tar",
        "imagenet-c/digital.tar",
        "imagenet-c/extra.tar",
        "imagenet-c/noise.tar",
        "imagenet-c/weather.tar",
    ],
)
def test_extract_dataset(name):
    import io
    import os
    import shutil
    import tarfile

    class ProgressFileObject(io.FileIO):
        def __init__(self, path, *args, **kwargs):
            self._total_size = os.path.getsize(path)
            io.FileIO.__init__(self, path, *args, **kwargs)
            self.pbar = tqdm(
                total=self._total_size,
                miniters=10000,
                unit="B",
                unit_scale=True,
                unit_divisor=1000,
                bar_format="        {percentage:3.0f}%|{bar}{r_bar}",
            )

        def read(self, size):
            delta = min(self.tell() + size, self._total_size) - self.pbar.n
            self.pbar.update(delta)
            return io.FileIO.read(self, size)

        def __del__(self):
            self.pbar.update(self.pbar.total - self.pbar.n)
            self.pbar.close()
            return super().__del__()

    path = os.path.join("datasets/", name)
    print(f"\n{path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        raise ValueError(
            'Archive not found "{path}". Make sure you downloaded the data first using "test_doanload_data".'
        )

    # Check md5sum
    check_md5sum(name, path)

    # Extract archive
    print("\tReading archive...")
    if name in to_untar:
        tar = tarfile.open(fileobj=ProgressFileObject(path))
        extract_folders = [n for n in tar.getnames() if "/" not in n]
        tar.close()
        del tar
        extract_path = os.path.dirname(path)
        for folder in extract_folders:
            p = os.path.join(extract_path, folder)
            if os.path.exists(p):
                print(f'\tDeleting "{p}"')
                shutil.rmtree(p, ignore_errors=True)

        print(f"\tExtracting...")
        tar = tarfile.open(fileobj=ProgressFileObject(path))
        tar.extractall(extract_path)
        tar.close()
        del tar
        print("\tDONE.")
    else:
        print("\tSKIPPED: no need to untar.")


def test_extract_dataset_split(split):
    if split == "test_r":
        names = ["imagenet-r.tar"]
    elif "test_c" in split:
        names = [
            "imagenet-c/blur.tar",
            "imagenet-c/digital.tar",
            "imagenet-c/extra.tar",
            "imagenet-c/noise.tar",
            "imagenet-c/weather.tar",
        ]
    else:
        names = []

    for name in names:
        test_extract_dataset(name)


@pytest.mark.parametrize("severity", [5])
def test_make_imagenet_c_merged_no_rep(severity):
    """Create a dataset by drawing without repetition from the corrupted
    ImageNet-C datasets, stratified by corruption type.
    The resulting dataset contains 50000 images, each having one of the 19 corruptions.
    The stratification ensures that each of the 19 corruption type has equally many images
    (up to a rounding factor).
    """
    import shutil

    corruptions = [
        "snow",
        "spatter",
        "shot_noise",
        "speckle_noise",
        "zoom_blur",
        "saturate",
        "pixelate",
        "motion_blur",
        "jpeg_compression",
        "impulse_noise",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "frost",
        "fog",
        "elastic_transform",
        "defocus_blur",
        "brightness",
        "contrast",
    ]

    n_corruptions = len(corruptions)

    net = IMAGENET_AlexNet(split="val")
    ds = net.get_dataset()
    n_samples = len(ds)

    q = n_samples // n_corruptions
    r = n_samples % n_corruptions
    n_samples_per_corruption = np.full(n_corruptions, q, dtype=int)
    n_samples_per_corruption[:r] += 1

    assert np.all(n_samples_per_corruption > 0)
    assert np.sum(n_samples_per_corruption) == n_samples

    corruption_idx = np.repeat(np.arange(n_corruptions), n_samples_per_corruption)

    assert len(corruption_idx) == n_samples

    rs = np.random.RandomState(0)
    corruption_idx = rs.permutation(corruption_idx)

    uniques, counts = np.unique(corruption_idx, return_counts=True)

    assert np.sum(counts) == n_samples
    assert np.array_equal(uniques, np.arange(n_corruptions))

    root = join(ds.root, "val")

    out_rootpath = f"datasets/imagenet-c/_merged_no_rep/{severity}/"

    print(f"\n{out_rootpath}")
    print(f"\tMerging corruptions (severity {severity})...")
    for i, (imgpath, _) in enumerate((pbar := tqdm(ds.imgs))):
        relpath = os.path.relpath(imgpath, root)
        corruption_id = corruption_idx[i]
        corruption_name = corruptions[corruption_id]

        in_imgpath = join(f"datasets/imagenet-c/{corruption_name}/{severity}/", relpath)
        subfolders, imgname = os.path.split(relpath)
        out_dirpath = join(out_rootpath, subfolders)
        os.makedirs(out_dirpath, exist_ok=True)
        out_imgpath = join(out_dirpath, f"{corruption_name}_{imgname}")

        shutil.copyfile(in_imgpath, out_imgpath)
    print("\tDONE.")


def test_make_datasets(split=None):
    if split is None:  # make all
        test_download_datasets()
        for name in urls.keys():
            test_extract_dataset(name)
        test_make_imagenet_c_merged_no_rep(5)

    elif split in ["test_seen", "test_unseen"]:  # make nlp only
        test_download_nlp_network(split, 0)

    else:  # make vision split only
        test_download_vision_datasets(splits=[split])
        test_extract_dataset_split(split=split)
        if split == "test_c:_merged_no_rep5":
            test_make_imagenet_c_merged_no_rep(severity=5)


def get_network(net: BaseNet, split: str, best: bool):
    if best:
        version = best_versions.get(net, None)
        if version is None:
            pytest.skip(f'No best version "{version}" found for net "{net}"')
        net = net(split=split, type=version)  # best version
    else:
        net = net(split=split)  # default version (ie smallest)
    return net


@pytest.mark.parametrize("net", nets)
@pytest.mark.parametrize("best", bests)
def test_download_vision_networks(net: BaseNet, best: bool):
    # This will automatically trigger download if necessary
    get_network(net=net, split="val", best=best)


@pytest.mark.parametrize(
    "split",
    [
        "test_seen",
        "test_unseen",
    ],
)
def test_download_nlp_network(split, worker_id):
    # This will automatically trigger download if necessary
    net = ZeroShotBartYahoo(split=split)
    net.get_dataset()  # Also triggers download of dataset


def id_from_worker_id(worker_id):
    worker_id = str(worker_id)
    r = re.match(r"^(master|(gw)?([0-9]*))$", worker_id)
    if r.group(3) is not None:
        i = int(r.group(3))
    elif r.group(1) == "master":
        i = 0
    else:
        raise ValueError(f'worker_id "{worker_id}" not understood.')
    return i


@pytest.mark.parametrize("net", nets)
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("best", bests)
def test_forward_vision_networks(
    net: BaseNet, split: str, best: bool, worker_id: str, n_jobs: int
):
    net = get_network(net=net, split=split, best=best)
    position = id_from_worker_id(worker_id)
    net.forward_dataset(skip_exist=True, verbose=True, position=position)
    net.load_forwarded_dataset(
        skip_exist=True, verbose=True, position=position, n_jobs=n_jobs
    )


@pytest.mark.parametrize(
    "split",
    [
        "test_seen",
        "test_unseen",
    ],
)
def test_forward_nlp_network(split, worker_id, n_jobs):
    net = ZeroShotBartYahoo(split=split)
    position = id_from_worker_id(worker_id)
    net.forward_dataset(skip_exist=True, verbose=True, position=position)
    net.load_forwarded_dataset(
        skip_exist=True, verbose=True, position=position, n_jobs=n_jobs
    )


def _requirement_base(func, _split, **kwargs):
    try:
        func(**kwargs)  # Try running the function

    except (FileNotFoundError, RuntimeError):  # Problem with data
        test_make_datasets(split=_split)  # Make datasets
        func(**kwargs)  # Retry


def test_fig6_requirement(n_jobs: int):
    def func():
        nets = [
            IMAGENET_ConvNeXt,
            IMAGENET_VisionTransformer,
        ]

        def one(net, worker_id):
            test_download_vision_networks(net=net, best=False)
            test_forward_vision_networks(
                net=net, split="test_r", best=False, worker_id=worker_id, n_jobs=1
            )

        Parallel(n_jobs=n_jobs)(delayed(one)(net, i) for i, net in enumerate(nets))

    _requirement_base(func, "test_r")


def test_fig7_requirement(n_jobs: int, split="test_r", best=False):
    def func():
        def one(net, worker_id):
            test_download_vision_networks(net=net, best=best)
            test_forward_vision_networks(
                net=net, split=split, best=best, worker_id=worker_id, n_jobs=1
            )

        Parallel(n_jobs=n_jobs)(
            delayed(one)(net, i) for i, net in enumerate(ALL_IMAGENET_NETWORKS)
        )

    _requirement_base(func, split)


@pytest.mark.parametrize(
    "split",
    [
        "test_seen",
        "test_unseen",
    ],
)
def test_fig8_requirement(split, n_jobs, worker_id):
    _requirement_base(
        test_forward_nlp_network, split, split=split, worker_id=worker_id, n_jobs=n_jobs
    )


def test_fig13_requirement(n_jobs: int):
    test_fig7_requirement(n_jobs)


def test_fig14_requirement(n_jobs: int):
    def func():
        def one(net, split, best, worker_id):
            test_download_vision_networks(net=net, best=False)
            test_forward_vision_networks(
                net=net, split=split, best=best, worker_id=worker_id, n_jobs=1
            )

        Parallel(n_jobs=n_jobs)(
            delayed(one)(net, split, best, i)
            for i, (net, split, best) in enumerate(
                itertools.product(ALL_IMAGENET_NETWORKS, splits, bests)
            )
        )

    _requirement_base(func, None)


def test_fig15_requirement(n_jobs: int):
    test_fig7_requirement(n_jobs=n_jobs, best=False)


def test_fig16_requirement(n_jobs: int):
    test_fig15_requirement(n_jobs=n_jobs)


def test_fig17_requirement(n_jobs: int):
    test_fig15_requirement(n_jobs=n_jobs)


def test_fig18_requirement(n_jobs: int):
    test_fig15_requirement(n_jobs=n_jobs)


def test_fig19_requirement(n_jobs: int):
    test_fig7_requirement(n_jobs=n_jobs, best=True)


def test_fig20_requirement(n_jobs: int):
    test_fig7_requirement(n_jobs=n_jobs, split="test_c:_merged_no_rep5")


def test_fig21_requirement(n_jobs: int):
    test_fig7_requirement(n_jobs=n_jobs, split="test_c:_merged_no_rep5", best=True)


def test_fig22_requirement(n_jobs: int):
    test_fig20_requirement(n_jobs=n_jobs)


def test_fig23_requirement(n_jobs: int):
    test_fig21_requirement(n_jobs=n_jobs)


def test_fig24_requirement(n_jobs: int):
    test_fig7_requirement(n_jobs=n_jobs, split="val")


def test_fig25_requirement(n_jobs: int):
    test_fig7_requirement(n_jobs=n_jobs, split="val", best=True)


def test_fig26_requirement(n_jobs: int):
    test_fig24_requirement(n_jobs=n_jobs)


def test_fig27_requirement(n_jobs: int):
    test_fig25_requirement(n_jobs=n_jobs)
