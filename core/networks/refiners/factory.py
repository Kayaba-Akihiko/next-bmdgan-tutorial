#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import torch.nn as nn
from .upsample_refiner import P2PUpsampleRefiner
from typing import Optional

_ARCH_CLASS_TABLE = dict(
    p2p_upsampler_refiner=P2PUpsampleRefiner,
)


class RefinerFactory:
    supported_archs = tuple(_ARCH_CLASS_TABLE.keys())
    default_arch = 'p2p_upsampler_refiner'

    @classmethod
    def get_model(
            cls,
            in_channels: int,
            out_channels: int,
            arch: Optional[str] = None,
            **kwargs,
    ) -> nn.Module:
        if arch is None:
            arch = cls.default_arch
        model_class = _ARCH_CLASS_TABLE[arch]
        model = model_class(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
        return model
