#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from mmseg.registry import MODELS
import torch
import torch.nn as nn
from typing import Tuple


class FPNDecoder(nn.Module):

    def __init__(
            self,
            in_channels: Tuple[int, int, int, int],
            out_channels: int,
            dropout_ratio=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        neck_out_channels = max(in_channels[0], out_channels)
        neck = MODELS.build(dict(
            type='FPN',
            in_channels=list(in_channels),
            out_channels=neck_out_channels,
            num_outs=4
        ))
        decode_head = MODELS.build(dict(
            type='FPNHead',
            in_channels=[
                neck_out_channels,
                neck_out_channels,
                neck_out_channels,
                neck_out_channels,
            ],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=max(neck_out_channels // 2, out_channels),
            dropout_ratio=dropout_ratio,
            num_classes=out_channels,
            norm_cfg=dict(type='IN', requires_grad=True),
            align_corners=False,
        ))
        self.neck = neck
        self.decode_head = decode_head

        # self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.neck(x)
        x = self.decode_head(x)
        return x

    def init_weights(self):
        self.neck.init_weights()
        self.decode_head.init_weights()