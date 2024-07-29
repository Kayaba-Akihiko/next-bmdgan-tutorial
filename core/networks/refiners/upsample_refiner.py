#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import torch.nn as nn
import torch
from mmengine.model import BaseModule
import logging

_logger = logging.getLogger(__name__)


class P2PUpsampleRefiner(BaseModule):
    # Pix2Pix/CycleGAN Upsampler

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs,
    ):
        init_cfg = dict(
            type='Normal',
            std=0.001,
            layer=['Conv2d', 'ConvTranspose2d'])
        super(P2PUpsampleRefiner, self).__init__(init_cfg=init_cfg)

        if len(kwargs) > 0:
            _logger.warning(f'getting unused parameters: {kwargs}')

        n_downsampling = 2
        model = []

        in_channels_ = in_channels
        for i in range(n_downsampling):  # add upsampling layers
            out_channels_ = max(
                out_channels,
                int(in_channels / 2 ** (i + 1))
            )
            model += [
                nn.ConvTranspose2d(
                    in_channels_,
                    out_channels_,
                    kernel_size=3, stride=2,
                    padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels_),
                nn.ReLU(True)
            ]
            in_channels_ = out_channels_
        in_channels_ = max(
            out_channels,
            int(in_channels / 2 ** n_downsampling)
        )
        model += [nn.Conv2d(
            int(in_channels_),
            out_channels,
            kernel_size=7,
            padding=3,
            padding_mode='reflect',
        )]
        self.model = nn.Sequential(*model)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
