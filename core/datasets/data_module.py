#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from torch.utils.data import DataLoader
from typing import Optional
from utils.typing import TypeConfig
from utils import os_utils, import_utils
import logging
import platform

_logger = logging.getLogger(__name__)


class DataModule:

    def __init__(
            self,
            training_batch_size: Optional[int] = None,
            test_batch_size: Optional[int] = None,
            training_dataset_config: Optional[TypeConfig] = None,
            test_dataset_config: Optional[TypeConfig] = None,
            visualization_dataset_config: Optional[TypeConfig] = None,
            n_training_loading_workers: Optional[int] = None,
            n_test_loading_workers: Optional[int] = None,
    ):
        if training_batch_size is None:
            training_batch_size = 1
        elif training_batch_size < 1:
            training_batch_size = None
        else:
            pass

        if test_batch_size is None:
            test_batch_size = training_batch_size
        elif test_batch_size < 1:
            test_batch_size = None
        else:
            pass

        if n_training_loading_workers is None:
            if training_batch_size is not None:
                n_training_loading_workers = training_batch_size
            else:
                n_training_loading_workers = 0
        n_training_loading_workers = min(
            n_training_loading_workers, os_utils.get_max_n_worker())

        if n_test_loading_workers is None:
            if test_batch_size is not None:
                n_test_loading_workers = test_batch_size
            else:
                n_test_loading_workers = 0
        n_test_loading_workers = min(
            n_test_loading_workers, os_utils.get_max_n_worker())


        self._training_batch_size = training_batch_size
        self._test_batch_size = test_batch_size

        self._n_training_loading_workers = n_training_loading_workers
        self._n_test_loading_workers = n_test_loading_workers

        self._training_data_loader = None
        self._test_data_loader = None
        self._visual_data_loader = None

        _logger.info(
            f"Using training batch size: {self._training_batch_size}.")
        _logger.info(
            f"Using test batch size: {self._test_batch_size}.")

        pin_memory = True
        if platform.system() == "Windows":
            pin_memory = False

        if training_dataset_config is not None:
            _logger.info("Initializing training dataset.")
            training_dataset = import_utils.get_object_from_config(
                training_dataset_config,)
            training_data_loader = DataLoader(
                training_dataset,
                batch_size=self._training_batch_size,
                shuffle=True,
                num_workers=self._n_training_loading_workers,
                pin_memory=pin_memory,
            )
            self._training_data_loader = training_data_loader

        if test_dataset_config is not None:
            _logger.info("Initializing test dataset.")
            test_dataset = import_utils.get_object_from_config(
                test_dataset_config, )
            test_data_loader = DataLoader(
                test_dataset,
                batch_size=self._test_batch_size,
                shuffle=True,
                num_workers=self._n_test_loading_workers,
                pin_memory=pin_memory,
            )
            self._test_data_loader = test_data_loader
            if visualization_dataset_config is not None:
                _logger.info("Initializing visualization dataset.")
                from .default_dataset import VisualizationDataset
                visual_dataset = VisualizationDataset(
                    test_dataset=test_dataset,
                    **visualization_dataset_config
                )
                visual_data_loader = DataLoader(
                    visual_dataset,
                    batch_size=self._test_batch_size,
                    shuffle=True,
                    num_workers=self._n_test_loading_workers,
                    pin_memory=pin_memory,
                )
                self._visual_data_loader = visual_data_loader

    @property
    def training_data_loader(self) -> Optional[DataLoader]:
        return self._training_data_loader

    @property
    def test_data_loader(self) -> Optional[DataLoader]:
        return self._test_data_loader

    @property
    def visualization_data_loader(self) -> Optional[DataLoader]:
        return self._visual_data_loader

    def set_epoch(self, epoch: int) -> None:
        if self._visual_data_loader is not None:
            self._visual_data_loader.dataset.reset_pool()