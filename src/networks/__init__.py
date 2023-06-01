from .base import BaseNet
from .imagenet import VGG as IMAGENET_VGG
from .imagenet import AlexNet as IMAGENET_AlexNet
from .imagenet import ConvNeXt as IMAGENET_ConvNeXt
from .imagenet import DenseNet as IMAGENET_DenseNet
from .imagenet import EfficientNet as IMAGENET_EfficientNet
from .imagenet import GoogLeNet as IMAGENET_GoogLeNet
from .imagenet import Inception as IMAGENET_Inception
from .imagenet import MNASNet as IMAGENET_MNASNet
from .imagenet import MobileNet as IMAGENET_MobileNet
from .imagenet import RegNet as IMAGENET_RegNet
from .imagenet import ResNet as IMAGENET_ResNet
from .imagenet import ResNext as IMAGENET_ResNext
from .imagenet import ShuffleNet as IMAGENET_ShuffleNet
from .imagenet import VisionTransformer as IMAGENET_VisionTransformer
from .imagenet import WideResNet as IMAGENET_WideResNet
from .yahoo import ZeroShotBartYahoo

ALL_IMAGENET_NETWORKS = [
    IMAGENET_VGG,
    IMAGENET_ResNet,
    IMAGENET_AlexNet,
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
    IMAGENET_ConvNeXt,
]

ALL_YAHOO_NETWORKS = [
    ZeroShotBartYahoo,
]

ALL_NETWORKS = ALL_IMAGENET_NETWORKS + ALL_YAHOO_NETWORKS
