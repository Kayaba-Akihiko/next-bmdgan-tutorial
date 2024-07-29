#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import Protocol, runtime_checkable
from typing import List, Dict, AnyStr, Union, Optional, Tuple
from abc import abstractmethod
from utils.typing import TypePathLike, TypeNPDTypeFloat
from torch.optim import Optimizer
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from pathlib import Path


@runtime_checkable
class ModelProtocol(Protocol):
    _image_size: Tuple[int, int]
    _image_channels: int
    _device: torch.device
    _networks: Dict[str, torch.nn.Module]

    @abstractmethod
    def load_model(
            self, load_dir: Path, prefix='ckp', strict=True,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def trigger_model(self, train: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def _register_network(
            self, name: str, net: torch.nn.Module) -> None:
        raise NotImplementedError


@runtime_checkable
class TrainingModelProtocol(ModelProtocol, Protocol):

    _grad_scalers: List[torch.amp.GradScaler]

    @property
    @abstractmethod
    def optimizers(self)  -> List[Optimizer]:
        raise NotImplementedError


    @abstractmethod
    def train_batch(
            self,
            data: Dict,
            epoch: int
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_epoch(
            self, test_data_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def visualize_epoch(
            self, visualization_data_loader: DataLoader
    ) -> Dict[str, Union[torch.Tensor, npt.NDArray[TypeNPDTypeFloat]]]:
        raise NotImplementedError

    @abstractmethod
    def save_model(self, save_dir: Path, prefix='ckp') -> None:
        raise NotImplementedError


@runtime_checkable
class TestModelProtocol(ModelProtocol, Protocol):
    @abstractmethod
    def test_and_save(
            self,
            test_data_loader: DataLoader,
            save_dir: Path,
    ) -> None:
        raise NotImplementedError