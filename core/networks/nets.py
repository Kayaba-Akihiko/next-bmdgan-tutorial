#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.
import copy
import logging
import functools
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Tuple, Union, Dict, Any, Optional, Self, Protocol, Iterator, Mapping

from utils import container_utils, torch_utils
from utils.typing import TypePathLike
from .backbones.factory import BackboneFactory
from .decoders.factory import DecoderFactory
from .refiners.factory import RefinerFactory
from .classifier_heads.factory import ClassifierHeadFactory
from pathlib import Path
import re


_logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    supported_backbones = BackboneFactory.supported_archs
    def __init__(
            self,
            in_channels: int,
            image_size: Union[Tuple[int, int], int],
            backbone_cfg: Dict[str, Any],
            pretrain_backbone_load_path: Optional[TypePathLike] = None,
            pretrain_backbone_strict_load=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        image_size = container_utils.to_2tuple(image_size)
        self.image_size = image_size
        res = BackboneFactory.get_model(
            in_channels=self.in_channels,
            image_size=self.image_size,
            **backbone_cfg,
        )
        self.backbone_name = backbone_cfg['arch']
        self.backbone = res['model']
        self.backbone_out_channels = res['out_channels']

        if pretrain_backbone_load_path is not None:
            _logger.info(
                f'Initializing backbone from {pretrain_backbone_load_path}.'
            )
            self.load_pretrain_backbone(
                pretrain_backbone_load_path,
                pretrain_backbone_strict_load,
            )
            _logger.info(
                f'Pretrained backbone loaded from '
                f'{pretrain_backbone_load_path}.'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x

    def load_pretrain_backbone(
            self, load_path: TypePathLike, strict: bool = True):
        load_path = Path(load_path)
        state_dict = torch.load(load_path, weights_only=True)
        backbone_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                k = k[9:]
                backbone_state_dict[k] = v
        if self.backbone_name.startswith('swinv2') and strict:
            ignore_keys_re = re.compile(
                '.+relative_(position_index|coords_table)$')
            missing, unexpected = torch_utils.load_network_by_dict(
                self.backbone,
                backbone_state_dict,
                log=False,
                strict=False,
            )
            if len(unexpected) > 0:
                raise RuntimeError(
                    f'Unexpected keys in backbone: {unexpected}.')

            if len(missing) > 0:
                allow_missing = True
                for key in missing:
                    if ignore_keys_re.match(key) is None:
                        allow_missing = False
                        break
                if not allow_missing:
                    raise RuntimeError(
                        f'Missing keys in backbone: {missing}.'
                    )
        else:
            torch_utils.load_network_by_dict(
                self.backbone,
                backbone_state_dict,
                log=True,
                strict=strict,
            )

    def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict: bool = True,
            assign: bool = False,
    ):
        if self.backbone_name.startswith('swinv2') and strict:
            ignore_keys_re = re.compile(
                '.+relative_(position_index|coords_table)$')
            missing, unexpected = super().load_state_dict(
                state_dict=state_dict,
                strict=False,
                assign=assign,
            )
            if len(unexpected) > 0:
                raise RuntimeError(
                    f'Unexpected keys in backbone: {unexpected}.')
            if len(missing) > 0:
                allow_missing = True
                for key in missing:
                    if ignore_keys_re.match(key) is None:
                        allow_missing = False
                        break
                if not allow_missing:
                    raise RuntimeError(
                        f'Missing keys in backbone: {missing}.'
                    )
            return missing, unexpected
        else:
            return super().load_state_dict(
                state_dict=state_dict,
                strict=strict,
                assign=assign,
            )



class Classifier(Encoder):
    def __init__(
            self,
            in_channels: int,
            image_size: Union[Tuple[int, int], int],
            n_classes: int,
            backbone_cfg: Dict[str, Any],
            head_cfg: Optional[Dict[str, Any]] = None,
            pretrain_backbone_load_path: Optional[TypePathLike] = None,
            pretrain_backbone_strict_load=True,
            use_mlp=False,
    ):
        backbone_cfg = copy.deepcopy(backbone_cfg)
        if backbone_cfg['arch'].startswith('convnextv2'):
            if 'gap_before_final_norm' in backbone_cfg:
                _logger.info(
                    'Overriding gap_before_final_norm to True in backbone.')
            backbone_cfg['gap_before_final_norm'] = True
        super().__init__(
            in_channels=in_channels,
            image_size=image_size,
            backbone_cfg=backbone_cfg,
            pretrain_backbone_load_path=pretrain_backbone_load_path,
            pretrain_backbone_strict_load=pretrain_backbone_strict_load,
        )
        self.n_classes = n_classes

        if head_cfg is None:
            head_cfg = {'pooling': 'avg'}

        head_cfg = copy.deepcopy(head_cfg)
        if backbone_cfg['arch'].startswith('convnextv2'):
            head_cfg['pooling'] = 'none'
        if use_mlp:
            head_cfg['hidden_channels'] = self.backbone_out_channels[-1]

        head = ClassifierHeadFactory.get_model(
            in_channels=self.backbone_out_channels[-1],
            out_channels=self.n_classes,
            **head_cfg,
        )
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)[-1]
        x = self.head(x)
        return x


class ContrastiveLearningEncoder(Classifier):
    def __init__(
            self,
            in_channels: int,
            image_size: Union[Tuple[int, int], int],
            backbone_cfg: Dict[str, Any],
            projection_dim=128,
            pretrain_backbone_load_path: Optional[TypePathLike] = None,
            pretrain_backbone_strict_load=True,
            use_mlp=True,
    ):
        super().__init__(
            in_channels=in_channels,
            image_size=image_size,
            n_classes=projection_dim,
            backbone_cfg=backbone_cfg,
            pretrain_backbone_load_path=pretrain_backbone_load_path,
            pretrain_backbone_strict_load=pretrain_backbone_strict_load,
            use_mlp=use_mlp
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = F.normalize(x, dim=1)
        return x


class Segmentor(Encoder):
    def __init__(
            self,
            in_channels: int,
            image_size: Union[Tuple[int, int], int],
            n_classes: int,
            backbone_cfg: Dict[str, Any],
            decoder_cfg: Optional[Dict[str, Any]] = None,
            pretrain_backbone_load_path: Optional[TypePathLike] = None,
            pretrain_backbone_strict_load=True,
    ):
        super().__init__(
            in_channels=in_channels,
            image_size=image_size,
            backbone_cfg=backbone_cfg,
            pretrain_backbone_load_path=pretrain_backbone_load_path,
            pretrain_backbone_strict_load=pretrain_backbone_strict_load,
        )
        self.n_classes = n_classes
        if decoder_cfg is None:
            decoder_cfg = {}

        decoder = DecoderFactory.get_model(
            in_channels=self.backbone_out_channels,
            out_channels=self.n_classes,
            **decoder_cfg
        )
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.decoder(x)
        return x


class Generator(Segmentor):
    def __init__(
            self,
            in_channels: int,
            image_size: Union[Tuple[int, int], int],
            out_channels: int,
            backbone_cfg: Dict[str, Any],
            decoder_cfg: Optional[Dict[str, Any]] = None,
            decoder_channels=256,
            refiner_cfg: Optional[Dict] = None,
            pretrain_backbone_load_path: Optional[TypePathLike] = None,
            pretrain_backbone_strict_load=True,
    ):
        super().__init__(
            in_channels=in_channels,
            image_size=image_size,
            n_classes=decoder_channels,
            backbone_cfg=backbone_cfg,
            decoder_cfg=decoder_cfg,
            pretrain_backbone_load_path=pretrain_backbone_load_path,
            pretrain_backbone_strict_load=pretrain_backbone_strict_load,
        )
        self.out_channels = out_channels

        if refiner_cfg is None:
            refiner_cfg = {}

        refiner = RefinerFactory.get_model(
            in_channels=decoder_channels,
            out_channels=out_channels,
            **refiner_cfg,
        )
        self.refiner = refiner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.refiner(x)
        return x


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# or
# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py#L17
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        ndf_ = max(ndf, input_nc)  # to avoid channels reduction when input_nc > ndf
        sequence = [
            nn.Conv2d(input_nc, ndf_, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        previous_ndf = ndf_
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            ndf_ = max(ndf * nf_mult, input_nc)
            sequence += [
                nn.Conv2d(previous_ndf, ndf_, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf_),
                nn.LeakyReLU(0.2, True)
            ]
            previous_ndf = ndf_

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        ndf_ = max(ndf * nf_mult, input_nc)
        sequence += [
            nn.Conv2d(previous_ndf, ndf_, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf_),
            nn.LeakyReLU(0.2, True)
        ]
        previous_ndf = ndf_

        sequence += [
            nn.Conv2d(previous_ndf, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)