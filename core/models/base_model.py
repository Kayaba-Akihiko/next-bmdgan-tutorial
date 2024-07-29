#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any

from .protocol import ModelProtocol, TrainingModelProtocol, TestModelProtocol
import torch
import logging
from pathlib import Path
from utils import torch_utils, container_utils

_logger = logging.getLogger(__name__)


class BaseModel(ABC, ModelProtocol):
    def __init__(
            self,
            image_size: Union[Tuple[int, int], int],
            image_channels: int,
            device=torch.device('cpu'),
    ):
        self._image_size = container_utils.to_2tuple(image_size)
        self._image_channels = image_channels
        self._device = device
        self._networks = {}

    def _register_network(
            self, name: str, net: torch.nn.Module) -> None:
        if name in self._networks:
            raise ValueError(f'Net {name} already registered.')

        self._networks[name] = net

    def trigger_model(self, train: bool) -> None:
        for net in self._networks.values():
            net.train(train)

    def load_model(
            self, load_dir: Path, prefix='ckp', strict=True, resume=True
    ) -> None:
        for name, net in self._networks.items():
            load_path = load_dir / f'{prefix}_{name}.pt'
            torch_utils.load_network_by_path(
                net, load_path, strict=strict)
            _logger.info(f'net {name} loaded from {load_path}.')


class BaseTrainingModel(BaseModel, ABC, TrainingModelProtocol):

    def save_model(self, save_dir: Path, prefix='ckp') -> None:
        for name, net in self._networks.items():
            save_path = save_dir / f'{prefix}_{name}.pt'
            torch.save(net.state_dict(), save_path)
            _logger.info(
                f'Model {name} weights saved to {save_path}.')

    def train_batch(
            self,
            data: Dict,
            epoch: int
    ) -> Dict[str, torch.Tensor]:
        res = self._compute_loss(data=data, epoch=epoch)
        losses = res['loss']
        log = res['log']

        for scaler, optim, loss in zip(
                self._grad_scalers, self.optimizers, losses):
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        return log

    @abstractmethod
    def _compute_loss(
            self,
            data: Dict[str, Any],
            epoch: int,
    ) -> Dict[
        str,
        Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]
    ]:
        raise NotImplementedError


class BaseTestModel(BaseModel, ABC, TestModelProtocol):
    pass