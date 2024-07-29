#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from mmpretrain.models.builder import MODELS
import torch.nn as nn
from typing import Tuple, Union, Dict, Any, List
from abc import ABC, abstractmethod
from .protocol import BackboneBuilderProtocol, TypeBuildResults
from mmpretrain.models.backbones.swin_transformer_v2 import SwinTransformerV2

from utils.container_utils import update_dict_


class BaseSwinV2Builder(ABC, BackboneBuilderProtocol):

    def _build_config(self) -> Dict[str, Any]:
        res = dict(
            type='SwinTransformerV2',
            img_size=self._image_size,
            in_channels=self._in_channels,
            drop_path_rate=0.2,
            out_indices=(0, 1, 2, 3)
        )

        if min(self._image_size) < 224:
            res['window_size'] = 4
        return res

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

class SwinV2TinyBuilder(BaseSwinV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg.update(dict(arch='tiny'))
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 96, 192, 384, 768


class SwinV2SmallBuilder(BaseSwinV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg.update(dict(arch='small'))
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 96, 192, 384, 768


class SwinV2BaseBuilder(BaseSwinV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg.update(dict(arch='base'))
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 128, 256, 512, 1024


class SwinV2LargeBuilder(BaseSwinV2Builder):

    def _build_config(self) -> Dict[str, Any]:
        cfg = super()._build_config()
        cfg.update(dict(arch='large'))
        return cfg

    @property
    def _out_channels(self) -> Tuple[int, int, int, int]:
        return 192, 384, 768, 1536
