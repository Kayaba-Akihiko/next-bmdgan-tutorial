#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import Tuple, Dict, Type, Union, Sequence

import torch
import torch.nn as nn

from .protocol import BackboneBuilderProtocol

from .swin_transformer_v2 import (
    SwinV2TinyBuilder,
    SwinV2SmallBuilder,
    SwinV2BaseBuilder,
    SwinV2LargeBuilder,
)
from .convnext_v2 import (
    ConvNextV2AttoBuilder,
    ConvNextV2FemtoBuilder,
    ConvNextV2PicoBuilder,
    ConvNextV2NanoBuilder,
    ConvNextV2TinyBuilder,
    ConvNextV2SmallBuilder,
    ConvNextV2BaseBuilder,
    ConvNextV2LargeBuilder,
    ConvNextV2XLargeBuilder,
    ConvNextV2HugeBuilder,
)

_ARCH_BUILDER_CLASS_TABLE: Dict[str, Type[BackboneBuilderProtocol]] = dict(
    swinv2_tiny=SwinV2TinyBuilder,
    swinv2_small=SwinV2SmallBuilder,
    swinv2_base=SwinV2BaseBuilder,
    swinv2_large=SwinV2LargeBuilder,

    convnextv2_atto=ConvNextV2AttoBuilder,
    convnextv2_femto=ConvNextV2FemtoBuilder,
    convnextv2_pico=ConvNextV2PicoBuilder,
    convnextv2_nano=ConvNextV2NanoBuilder,
    convnextv2_tiny=ConvNextV2TinyBuilder,
    convnextv2_small=ConvNextV2SmallBuilder,
    convnextv2_base=ConvNextV2BaseBuilder,
    convnextv2_large=ConvNextV2LargeBuilder,
    convnextv2_xlarge=ConvNextV2XLargeBuilder,
    convnextv2_huge=ConvNextV2HugeBuilder,
)


class BackboneFactory:

    supported_archs = tuple(_ARCH_BUILDER_CLASS_TABLE.keys())

    @staticmethod
    def get_model(
            arch: str, in_channels: int, image_size: Tuple[int, int], **kwargs
    ) -> Dict[
        str,
        Union[
            Union[torch.nn.Module, Tuple[int, int, int, int]],
            BackboneBuilderProtocol,
        ]
    ]:
        builder_class = _ARCH_BUILDER_CLASS_TABLE[arch]
        builder = builder_class(
            in_channels=in_channels, image_size=image_size, **kwargs)
        build_results = builder.build()
        model = build_results['model']
        out_channels = build_results['out_channels']
        return {
            'model': model,
            'out_channels': out_channels,
            'builder': builder,
        }