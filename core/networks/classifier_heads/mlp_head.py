#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import torch.nn as nn
import torch
from mmengine.model import BaseModule
from typing import Optional
from einops.layers.torch import Rearrange


class MLPHead(BaseModule):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: Optional[int] = None,
            pooling='none'
    ):
        super().__init__()

        if pooling == 'avg':
            pool_class = nn.AdaptiveAvgPool2d
        elif pooling == 'max':
            pool_class = nn.AdaptiveMaxPool2d
        elif pooling == 'none':
            pool_class = None
        else:
            raise ValueError(pooling)

        if pool_class is not None:
            pool = nn.Sequential(
                pool_class(1),
                Rearrange('b c h w -> b (c h w)')
            )
        else:
            pool = nn.Identity()

        if hidden_channels is not None:
            fc_in = nn.Linear(in_channels, hidden_channels)
            fc_out = nn.Linear(hidden_channels, out_channels)
        else:
            fc_in = nn.Identity()
            fc_out = nn.Linear(in_channels, out_channels)

        self.pool = pool
        self.fc_in = fc_in
        self.fc_out = fc_out

    def forward(self, x):
        x = self.pool(x)
        x = self.fc_in(x)
        x = self.fc_out(x)
        return x