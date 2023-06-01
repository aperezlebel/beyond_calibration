"""Define PyTorch networks pre-trained on ImageNet."""
import os
import re

import torch
import torchvision.models as models
from joblib import Memory
from torchvision import datasets
from torchvision import transforms as T

from .base import BaseNet

memory = Memory("joblib_cache")


class ImageNetAFolder(datasets.ImageFolder):
    """Load the ImageNetA dataset. This wrapper is needed as the ImageNetA
    classes are 200 subclasses of ImageNet1K. For the samples of ImageNetA to
    be used with networks trained on ImageNet1K, we need to make the samples
    idx match."""

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        subfolders_are_idx=False,
    ):
        self.is_valid_file = is_valid_file
        # Whether name of subfolders are class idx or not
        self.subfolders_are_idx = subfolders_are_idx
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory):
        classes, _class_to_idx = super().find_classes(directory)

        # Retrieve the class_to_idx of ImageNet1K
        ds = datasets.ImageNet("datasets", split="val")
        # Store wnids as in ImageNet class
        self.wnids = ds.wnids

        if self.subfolders_are_idx:
            return classes, _class_to_idx

        class_to_idx = {k: ds.wnid_to_idx[k] for k in _class_to_idx.keys()}
        # class_to_idx contains the 200 class names with their associated
        # id in ImageNet1K

        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self.is_valid_file is None or self.is_valid_file(path):
                        if target_class not in available_classes:
                            available_classes.add(target_class)

        if not available_classes:
            raise ValueError("No class having valid filepath samples found.")

        class_to_idx = {k: v for k, v in class_to_idx.items() if k in available_classes}

        return classes, class_to_idx


class ImageNetBased(BaseNet):
    def __init__(self, split="val"):
        self.split = split
        super().__init__()

    def get_transform(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize]
        )
        return transform

    def get_dataset(self):
        transform = self.get_transform()
        if self.split in ["train", "val"]:
            return datasets.ImageNet("datasets", split=self.split, transform=transform)

        r = re.match(r"test_([^:]*):?(.*)", self.split)

        if r is None:
            raise ValueError(f"Unknown split {self.split}.")

        is_valid_file = None
        subfolders_are_idx = False

        if r.group(1) == "a":
            ds_path = "datasets/imagenet-a/"

        elif r.group(1) == "r":
            ds_path = "datasets/imagenet-r/"
            sublabel = r.group(2)
            if sublabel:

                def is_valid_file(s):
                    return bool(re.match(f".*/{sublabel}_.*", s))

        elif r.group(1) == "o":
            ds_path = "datasets/imagenet-o/"

        elif r.group(1) == "v2.1":
            ds_path = "datasets/imagenetv2-matched-frequency-format-val"
            subfolders_are_idx = True

        elif r.group(1) == "v2.2":
            ds_path = "datasets/imagenetv2-threshold0.7-format-val"
            subfolders_are_idx = True

        elif r.group(1) == "v2.3":
            ds_path = "datasets/imagenetv2-top-images-format-val"
            subfolders_are_idx = True

        elif r.group(1) == "c":
            sublabel = r.group(2)
            subr = re.match("([^0-9]*)([0-9]*)", sublabel)
            sublabel = subr.group(1)
            level = subr.group(2)
            ds_path = f"datasets/imagenet-c/{sublabel}/{level}/"

        elif r.group(1) == "p":
            sublabel = r.group(2)
            ds_path = f"datasets/imagenet-p/{sublabel}"

        else:
            raise ValueError(f"Unknown split {self.split}.")

        return ImageNetAFolder(
            ds_path,
            transform=transform,
            is_valid_file=is_valid_file,
            subfolders_are_idx=subfolders_are_idx,
        )

    def get_dataset_name(self):
        return f"ILSVRC2012_img_{self.split}"

    def get_w(self):
        return self.last_layer.weight.detach()

    def get_intercept(self):
        return self.last_layer.bias.detach()

    def logits_to_scores(self, y_logits):
        return torch.nn.functional.softmax(y_logits, dim=1)

    def get_class_names(self):
        dataset = self.get_dataset()
        return dataset.wnids


class VGG(ImageNetBased):
    def __init__(self, type="11", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "11":
            return models.vgg11(pretrained=True)
        if self.type == "13":
            return models.vgg13(pretrained=True)
        if self.type == "16":
            return models.vgg16(pretrained=True)
        if self.type == "19":
            return models.vgg19(pretrained=True)
        if self.type == "11_bn":
            return models.vgg11_bn(pretrained=True)
        if self.type == "13_bn":
            return models.vgg13_bn(pretrained=True)
        if self.type == "16_bn":
            return models.vgg16_bn(pretrained=True)
        if self.type == "19_bn":
            return models.vgg19_bn(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.classifier[6]
        del model.classifier[6]
        return model, last_layer


class ResNet(ImageNetBased):
    def __init__(self, type="18", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "18":
            return models.resnet18(pretrained=True)
        if self.type == "34":
            return models.resnet34(pretrained=True)
        if self.type == "50":
            return models.resnet50(pretrained=True)
        if self.type == "101":
            return models.resnet101(pretrained=True)
        if self.type == "152":
            return models.resnet152(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.fc
        model.fc = torch.nn.Identity()
        return model, last_layer


class AlexNet(ImageNetBased):
    def create_model(self):
        return models.alexnet(pretrained=True)

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.classifier[6]
        del model.classifier[6]
        return model, last_layer


class DenseNet(ImageNetBased):
    def __init__(self, type="121", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "121":
            return models.densenet121(pretrained=True)
        if self.type == "169":
            return models.densenet169(pretrained=True)
        if self.type == "161":
            return models.densenet161(pretrained=True)
        if self.type == "201":
            return models.densenet201(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.classifier
        model.classifier = torch.nn.Identity()
        return model, last_layer


class Inception(ImageNetBased):
    def create_model(self):
        return models.inception_v3(pretrained=True)

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.fc
        model.fc = torch.nn.Identity()
        return model, last_layer


class GoogLeNet(ImageNetBased):
    def create_model(self):
        return models.googlenet(pretrained=True)

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.fc
        model.fc = torch.nn.Identity()
        return model, last_layer


class ShuffleNet(ImageNetBased):
    def __init__(self, type="0_5", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "0_5":
            return models.shufflenet_v2_x0_5(pretrained=True)
        if self.type == "1_0":
            return models.shufflenet_v2_x1_0(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.fc
        model.fc = torch.nn.Identity()
        return model, last_layer


class MobileNet(ImageNetBased):
    def __init__(self, type="v2", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "v2":
            return models.mobilenet_v2(pretrained=True)
        if self.type == "v3L":
            return models.mobilenet_v3_large(pretrained=True)
        if self.type == "v3S":
            return models.mobilenet_v3_small(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        if self.type in ["v3L", "v3S"]:
            idx = 3
        elif self.type == "v2":
            idx = 1
        last_layer = model.classifier[idx]
        del model.classifier[idx]

        return model, last_layer


class ResNext(ImageNetBased):
    def __init__(self, type="50", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "50":
            return models.resnext50_32x4d(pretrained=True)
        if self.type == "101":
            return models.resnext101_32x8d(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.fc
        model.fc = torch.nn.Identity()
        return model, last_layer


class WideResNet(ImageNetBased):
    def __init__(self, type="50", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "50":
            return models.wide_resnet50_2(pretrained=True)
        if self.type == "101":
            return models.wide_resnet101_2(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.fc
        model.fc = torch.nn.Identity()
        return model, last_layer


class MNASNet(ImageNetBased):
    def __init__(self, type="0_5", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "0_5":
            return models.mnasnet0_5(pretrained=True)
        if self.type == "1_0":
            return models.mnasnet1_0(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.classifier[1]
        del model.classifier[1]
        return model, last_layer


class EfficientNet(ImageNetBased):
    def __init__(self, type="b0", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        if self.type == "b0":
            return models.efficientnet_b0(pretrained=True)
        if self.type == "b1":
            return models.efficientnet_b1(pretrained=True)
        if self.type == "b2":
            return models.efficientnet_b2(pretrained=True)
        if self.type == "b3":
            return models.efficientnet_b3(pretrained=True)
        if self.type == "b4":
            return models.efficientnet_b4(pretrained=True)
        if self.type == "b5":
            return models.efficientnet_b5(pretrained=True)
        if self.type == "b6":
            return models.efficientnet_b6(pretrained=True)
        if self.type == "b7":
            return models.efficientnet_b7(pretrained=True)
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.classifier[1]
        del model.classifier[1]
        return model, last_layer


class RegNet(ImageNetBased):
    def __init__(self, type="y_400mf", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        choices = {
            "y_400mf": models.regnet_y_400mf(pretrained=True),
            "y_800mf": models.regnet_y_800mf(pretrained=True),
            "y_1_6gf": models.regnet_y_1_6gf(pretrained=True),
            "y_3_2gf": models.regnet_y_3_2gf(pretrained=True),
            "y_8gf": models.regnet_y_8gf(pretrained=True),
            "y_16gf": models.regnet_y_16gf(pretrained=True),
            "y_32gf": models.regnet_y_32gf(pretrained=True),
            "x_400mf": models.regnet_x_400mf(pretrained=True),
            "x_800mf": models.regnet_x_800mf(pretrained=True),
            "x_1_6gf": models.regnet_x_1_6gf(pretrained=True),
            "x_3_2gf": models.regnet_x_3_2gf(pretrained=True),
            "x_8gf": models.regnet_x_8gf(pretrained=True),
            "x_16gf": models.regnet_x_16gf(pretrained=True),
            "x_32gf": models.regnet_x_32gf(pretrained=True),
        }
        if self.type in choices:
            return choices[self.type]
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.fc
        model.fc = torch.nn.Identity()
        return model, last_layer


class VisionTransformer(ImageNetBased):
    def __init__(self, type="b_16", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        choices = {
            "b_16": models.vit_b_16(pretrained=True),
            "b_32": models.vit_b_32(pretrained=True),
            "l_16": models.vit_l_16(pretrained=True),
            "l_32": models.vit_l_32(pretrained=True),
        }
        if self.type in choices:
            return choices[self.type]
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.heads.head
        model.heads.head = torch.nn.Identity()
        return model, last_layer


class ConvNeXt(ImageNetBased):
    def __init__(self, type="tiny", split="val"):
        self.type = type
        super().__init__(split)

    def create_model(self):
        choices = {
            "tiny": models.convnext_tiny(pretrained=True),
            "small": models.convnext_small(pretrained=True),
            "base": models.convnext_base(pretrained=True),
            "large": models.convnext_large(pretrained=True),
        }
        if self.type in choices:
            return choices[self.type]
        raise ValueError(
            f"Unknown version {self.type} for " f"{self.__class__.__name__.lower()}."
        )

    def create_truncated_model(self):
        model = self.create_model()
        last_layer = model.classifier[2]
        model.classifier[2] = torch.nn.Identity()
        return model, last_layer
