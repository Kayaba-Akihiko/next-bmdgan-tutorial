#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from torch.utils.data import Dataset
from .protocol import DatasetProtocol
import random
from typing import Any


class VisualizationDataset(Dataset, DatasetProtocol):

    def __init__(
            self,
            test_dataset: DatasetProtocol,
            pool_size: int
    ) -> None:
        self._test_dataset = test_dataset
        self._pool_size = pool_size
        self._idx_pool: list[int] = []

        if self._pool_size < 0:
            self._pool_size = len(test_dataset)
        elif self._pool_size > len(test_dataset):
            self._pool_size = len(test_dataset)

        self.reset_pool()

    def __len__(self) -> int:
        return self._pool_size

    def __getitem__(self, idx: int) -> Any:
        return self._test_dataset[self._idx_pool[idx]]

    def reset_pool(self) -> None:
        self._idx_pool = random.sample(
            range(len(self._test_dataset)), self._pool_size)