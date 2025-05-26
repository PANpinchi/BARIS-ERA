# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .convnext import ConvNeXt

# Different Fine-Tuning Methods
from .convnext_bitfit import ConvNeXt_bitfit
from .convnext_norm_tuning import ConvNeXt_norm_tuning
from .convnext_partial_1 import ConvNeXt_partial_1
from .convnext_conv_adapter import ConvNeXtWithConvAdapter
from .convnext_vpt import ConvNeXtWithVPT

from .swin_bitfit import SwinTransformer_bitfit
from .swin_norm_tuning import SwinTransformer_norm_tuning
from .swin_partial_1 import SwinTransformer_partial_1
from .swin_adapter import SwinTransformerWithAdapter
from .swin_lora import SwinTransformerWithLoRA
from .swin_adaptformer import SwinTransformerWithAdaptformer
from .swin_mona import SwinTransformerWithMona

# Our Methods
from .swin_ema import SwinTransformerWithEMA
from .convnext_ema import ConvNeXtWithEMA

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'SwinTransformerWithEMA', 'SwinTransformer_bitfit',
    'SwinTransformer_norm_tuning', 'SwinTransformer_partial_1',  'SwinTransformerWithAdapter',
    'SwinTransformerWithLoRA', 'SwinTransformerWithAdaptformer', 'SwinTransformerWithMona',
    'PyramidVisionTransformer', 'ConvNeXt', 'ConvNeXtWithEMA', 'ConvNeXt_bitfit',
    'ConvNeXt_norm_tuning', 'ConvNeXt_partial_1', 'ConvNeXtWithConvAdapter',
    'ConvNeXtWithVPT', 'PyramidVisionTransformerV2', 'EfficientNet'
]
