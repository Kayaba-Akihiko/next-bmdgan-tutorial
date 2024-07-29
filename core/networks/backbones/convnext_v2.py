#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from mmpretrain.models.builder import MODELS
from mmpretrain.models.backbones.convnext import ConvNeXt
import torch.nn as nn
from typing import Tuple, Union, Dict, Any, List
from abc import ABC, abstractmethod
from .protocol import BackboneBuilderProtocol, TypeBuildResults

from utils.container_utils import update_dict_


class BaseConvNeXtV2Builder(ABC, BackboneBuilderProtocol):

    def _build_config(self) -> Dict[str, Any]:
        return dict(
            type='ConvNeXt',
            in_channels=self._in_channels,
            drop_path_rate=0.1,
            layer_scale_init_value=0.,
            use_grn=True,
            gap_before_final_norm=False,  # True for classification
            out_indices=(0, 1, 2, 3),
        )

    @property
    @abstractmethod
    def _out_channels(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def build(self) -> TypeBuildResults:
        cfg = self._build_config()
        cfg = update_dict_(cfg, self._kwargs)
        model = MODELS.build(cfg)
        # model.init_weights()
        return dict(model=model, out_channels=self._out_channels)

    @staticmethod
    def get_layer_cam_layers(model: nn.Module) -> List[nn.Module]:
        layers = []
        out_indices = (0, 1, 2, 3)
        for i, stage in enumerate(model.stages):
            if i in out_indices:
                layers.append(stage[-1].depthwise_conv)
        return layers


class ConvNextV2AttoBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'atto'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 40, 80, 160, 320


class ConvNextV2FemtoBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'femto'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 48, 96, 192, 384


class ConvNextV2PicoBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'pico'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 64, 128, 256, 512


class ConvNextV2NanoBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'nano'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 80, 160, 320, 640


class ConvNextV2TinyBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'tiny'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 96, 192, 384, 768


class ConvNextV2SmallBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'small'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 96, 192, 384, 768


class ConvNextV2BaseBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'base'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 128, 256, 512, 1024


class ConvNextV2LargeBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'large'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 192, 384, 768, 1536


class ConvNextV2XLargeBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'xlarge'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 256, 512, 1024, 2048


class ConvNextV2HugeBuilder(BaseConvNeXtV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg['arch'] = 'huge'
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 352, 704, 1408, 2816