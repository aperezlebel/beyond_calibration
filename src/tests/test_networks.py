import tempfile

import pytest
import torch

from src.networks import (
    ALL_IMAGENET_NETWORKS,
    ALL_NETWORKS,
    ALL_YAHOO_NETWORKS,
    IMAGENET_VGG,
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
)


@pytest.fixture
def net(request):
    """Initialize network at test run time instead of collection time."""

    if isinstance(request.param, tuple):
        dataset, split, network, version = request.param
    else:
        dataset, split, network, version = None, None, request.param, None

    # Retrieve network by name if string given else take input as class
    if isinstance(network, str):
        if dataset is not None:
            dataset = dataset.lower()
        else:
            dataset = "imagenet"

        if dataset == "imagenet":
            NETWORKS = ALL_IMAGENET_NETWORKS

        elif dataset == "yahoo":
            NETWORKS = ALL_YAHOO_NETWORKS

        else:
            raise ValueError(
                f"Unknown dataset {dataset}. " f"Choices: imagenet, yahoo."
            )

        networks_by_name = {c.__name__.lower(): c for c in NETWORKS}

        if network in networks_by_name.keys():
            network = networks_by_name[network]
        else:
            raise ValueError(
                f"Unknown network {network} for dataset {dataset}. "
                f"Choices: {list(networks_by_name.keys())}."
            )

    kwargs = {}
    if version is not None:
        kwargs["type"] = version
    if split is not None:
        kwargs["split"] = split

    try:
        return network(**kwargs)

    except TypeError:
        raise TypeError(f"Can't instantiate given network {network}.")


@pytest.mark.parametrize("net", ALL_NETWORKS, indirect=True)
def test_truncated_followed_by_last_layer(net):
    """Test if forwarding the truncated network followed by the last layer
    is equivalent to forwarding the full network."""
    loader = net.get_loader(batch_size=1)
    it = next(iter(loader))
    input, _ = net.loader_iter_to_input_label(it)
    y1 = net.forward(input, truncated=False)
    Xt = net.forward(input, truncated=True)
    y2 = net.last_layer(Xt)

    assert Xt.ndim == 2
    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("net", ALL_NETWORKS, indirect=True)
def test_networks_are_in_eval_mode(net):
    """Test if networks are in evaluation mode."""
    assert not net.model.training
    assert not net.truncated_model.training
    assert not net.last_layer.training


@pytest.mark.parametrize("net", [IMAGENET_VGG], indirect=True)
def test_stacking_Xt_y_scores(net):
    """Test if loading by batch is equivalent to loading in one shot."""
    Xt1, y_scores1, y_labels1 = net._get_Xt_y_scores_labels.func(
        batch_size=1, n_batch_max=2
    )
    Xt2, y_scores2, y_labels2 = net._get_Xt_y_scores_labels.func(
        batch_size=2, n_batch_max=1
    )

    assert torch.allclose(Xt1, Xt2, atol=1e-6)
    assert torch.allclose(y_scores1, y_scores2, atol=1e-6)
    assert torch.allclose(y_labels1, y_labels2, atol=1e-6)


@pytest.mark.parametrize("net", ALL_NETWORKS, indirect=True)
def test_tensors_are_detached(net):
    w = net.get_w()
    b = net.get_intercept()
    Xt, y_scores, y_labels = net._get_Xt_y_scores_labels.func(
        batch_size=1, n_batch_max=1
    )

    assert not w.requires_grad
    assert not b.requires_grad
    assert not Xt.requires_grad
    assert not y_scores.requires_grad
    assert not y_labels.requires_grad


@pytest.mark.parametrize("net", ALL_IMAGENET_NETWORKS, indirect=True)
def test_forward_load(net):
    """Test if dumping a forwarded dataset followed by loading it is
    equivalent to forwarding it without dumping."""
    batch_size = 2
    n_batch_max = 5
    dir = tempfile.TemporaryDirectory()
    dirpath = dir.name

    net.forward_dataset(
        dirpath, batch_size=batch_size, n_batch_max=n_batch_max, verbose=1, n_jobs=2
    )
    Xt1, scores1, labels1 = net.load_forwarded_dataset(
        dirpath, batch_size=batch_size, n_batch_max=n_batch_max, verbose=1, n_jobs=2
    )

    loader = net.get_loader(batch_size=batch_size * n_batch_max, shuffle=False)
    it = next(iter(loader))
    input, labels2 = net.loader_iter_to_input_label(it)
    Xt2 = net.forward(input, truncated=True)
    scores2 = net.logits_to_scores(net.last_layer(Xt2)).detach()

    dir.cleanup()

    assert torch.allclose(labels1, labels2, atol=1e-6)
    assert torch.allclose(scores1, scores2, atol=1e-6)
    assert torch.allclose(Xt1, Xt2, atol=1e-5)


@pytest.mark.parametrize("net", ALL_NETWORKS, indirect=True)
def test_dumping_does_not_affect_dataset(net):
    """Test if dumping Xt, y_scores and y_labels affects future loadings."""
    batch_size = 1
    n_batch_max = 2
    dir = tempfile.TemporaryDirectory()
    dirpath = dir.name

    net.forward_dataset(
        dirpath, batch_size=batch_size, n_batch_max=n_batch_max, verbose=1, n_jobs=2
    )
    Xt1, scores1, labels1 = net.load_forwarded_dataset(
        dirpath,
        batch_size=batch_size,
        n_batch_max=n_batch_max,
        dump=True,
        verbose=1,
        n_jobs=2,
    )
    # Xt2, scores2, labels2 should be the same as previous call
    # even if Xt1, scores1, labels1 have been saved to dataset folder
    Xt2, scores2, labels2 = net.load_forwarded_dataset(
        dirpath,
        batch_size=batch_size,
        n_batch_max=n_batch_max,
        dump=False,
        verbose=1,
        n_jobs=2,
    )

    dir.cleanup()

    assert torch.allclose(labels1, labels2)
    assert torch.allclose(scores1, scores2)
    assert torch.allclose(Xt1, Xt2)


@pytest.mark.parametrize("net", ALL_NETWORKS, indirect=True)
def test_dump_names(net):
    """Test the naming convention."""
    s = str(net)
    print(s)
    dataset_name, model = str(net).split("@")
    model_name, model_version = (model.split(":") + [None])[:2]
    assert dataset_name == net.get_dataset_name()
    assert model_name == net.__class__.__name__.lower()
    if model_version is not None:
        assert model_version == str(net.type)


@pytest.mark.parametrize("type", ["11", "13", "16", "19"])
def test_init_vgg(type):
    IMAGENET_VGG(type=type)


@pytest.mark.parametrize("type", ["18", "34", "50", "101", "152"])
def test_init_resnet(type):
    IMAGENET_ResNet(type=type)


def test_init_alexnet():
    IMAGENET_AlexNet()


@pytest.mark.parametrize("type", ["121", "169", "161", "201"])
def test_init_densenet(type):
    IMAGENET_DenseNet(type=type)


def test_init_inception():
    IMAGENET_Inception()


def test_init_googlenet():
    IMAGENET_GoogLeNet()


@pytest.mark.parametrize("type", ["0_5", "1_0"])
def test_init_shufflenet(type):
    IMAGENET_ShuffleNet(type=type)


@pytest.mark.parametrize("type", ["v2", "v3L", "v3S"])
def test_init_mobilenet(type):
    IMAGENET_MobileNet(type=type)


@pytest.mark.parametrize("type", ["50", "101"])
def test_init_resnext(type):
    IMAGENET_ResNext(type=type)


@pytest.mark.parametrize("type", ["50", "101"])
def test_init_wideresnet(type):
    IMAGENET_WideResNet(type=type)


@pytest.mark.parametrize("type", ["0_5", "1_0"])
def test_init_mnasnet(type):
    IMAGENET_MNASNet(type=type)


@pytest.mark.parametrize("type", [f"b{i}" for i in range(8)])
def test_init_efficientnet(type):
    IMAGENET_EfficientNet(type=type)


@pytest.mark.parametrize(
    "type",
    [
        "y_400mf",
        "y_800mf",
        "y_1_6gf",
        "y_3_2gf",
        "y_8gf",
        "y_16gf",
        "y_32gf",
        "x_400mf",
        "x_800mf",
        "x_1_6gf",
        "x_3_2gf",
        "x_8gf",
        "x_16gf",
        "x_32gf",
    ],
)
def test_init_regnet(type):
    IMAGENET_RegNet(type=type)


@pytest.mark.parametrize("type", ["b_16", "b_32", "l_16", "l_32"])
def test_init_visiontransformer(type):
    IMAGENET_VisionTransformer(type=type)


@pytest.mark.parametrize("type", ["tiny", "small", "base", "large"])
def test_init_convnext(type):
    IMAGENET_ConvNeXt(type=type)


@pytest.mark.parametrize("split", ["train", "val", "test_a"])
def test_imagenet_splits(split):
    IMAGENET_AlexNet(split=split)
