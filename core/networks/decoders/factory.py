#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import Tuple, Optional
import torch.nn as nn
from .fpn_decoder import FPNDecoder


_ARCH_CLASS_TABLE = dict(
    fpn_decoder=FPNDecoder,
)


class DecoderFactory:

    supported_archs = tuple(_ARCH_CLASS_TABLE.keys())
    default_arch = 'fpn_decoder'

    @classmethod
    def get_model(
            cls,
            in_channels: Tuple[int, int, int, int],
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
