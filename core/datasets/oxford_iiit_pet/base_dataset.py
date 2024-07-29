#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from utils import os_utils
from utils.typing import TypePathLike
from utils.multiprocess_agent import MultiProcessingAgent
import imageio.v3 as iio
from skimage.util import random_noise
from pathlib import Path
from typing import Tuple, Union, Optional, List
import logging
from torchvision.datasets.utils import (
    download_and_extract_archive, verify_str_arg
)
import numpy as np
from ..protocol import DatasetProtocol
from .protocol import SampleProtocol


_logger = logging.getLogger(__name__)


def _load_sample(sample: SampleProtocol) -> SampleProtocol:
    return sample.load()


class BaseDataset(Dataset, ABC, DatasetProtocol):

    def __init__(
            self,
            data_root: TypePathLike,
            n_workers=None,
    ):
        if n_workers is None:
            n_workers = os_utils.get_max_n_worker()
        else:
            n_workers = min(n_workers, os_utils.get_max_n_worker())
        self._n_workers = n_workers

        self._data_root = os_utils.format_path_str(data_root)
        self._images_folder = self._data_root / 'images'
        self._noised_images_folder = self._data_root / 'noised_images'
        self._anns_folder = self._data_root / 'annotations'
        self._trimaps_folder = self._anns_folder / 'trimaps'
        self._xmls_folder = self._anns_folder / 'xmls'


        self._norm_mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32)
        self._norm_std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)

        # self._norm_mean = self._norm_mean[np.newaxis, np.newaxis, ...]
        # self._norm_std = self._norm_std[np.newaxis, np.newaxis, ...]

        exists, folder = self._check_resources()
        if not exists:
            _logger.info(
                f'Could not find resources {folder}. '
                f'Proceeding downloading.'
            )
            # _download_dataset(self._data_root)
            _add_noise_to_images(
                self._images_folder,
                self._noised_images_folder,
                n_workers=self._n_workers
            )

        self._pool: List[SampleProtocol] = []

    def _preload_pool(self, desc: str):
        iterator = MultiProcessingAgent().run(
            args=[(sample, ) for sample in self._pool],
            func=_load_sample,
            n_workers=self._n_workers,
            desc=desc,
            mininterval=30, maxinterval=60,
        )
        pool = list(iterator)
        self._pool = pool

    def __len__(self) -> int:
        return len(self._pool)


    def _check_resources(self) -> Tuple[bool, Optional[Path]]:
        for folder in [
            self._images_folder,
            self._noised_images_folder,
            self._anns_folder,
        ]:
            if not folder.exists():
                return False, folder
        return True, None

def _download_dataset(
        output_dir: TypePathLike,
):
    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
         "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
         "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )

    for url, md5 in _RESOURCES:
        download_and_extract_archive(
            url, download_root=str(output_dir), md5=md5)


def _add_noise_to_images(
        images_folder: TypePathLike,
        noised_images_folder: TypePathLike,
        gaussian_var=0.10,
        n_workers=None,
):
    if n_workers is None:
        n_workers = os_utils.get_max_n_worker()
    else:
        n_workers = min(n_workers, os_utils.get_max_n_worker())

    args = []
    for file_entry in os_utils.scan_dirs_for_file(
            images_folder, '.+\\.jpg$'):
        load_path = Path(file_entry.path)
        file_name = load_path.name
        save_path = noised_images_folder / file_name
        args.append((load_path, save_path, gaussian_var))
    noised_images_folder.mkdir(parents=True, exist_ok=True)
    iterator = MultiProcessingAgent().run(
        args=args,
        func=_add_noise_to_image,
        n_workers=n_workers,
        desc='Adding noise'
    )
    for _ in iterator:
        pass

def _add_noise_to_image(
        load_path: TypePathLike,
        save_path: TypePathLike,
        gaussian_var=0.10,
):
    image = iio.imread(load_path)
    image = image[..., :3]
    image = image.astype(float) / 255.
    image = random_noise(
        image, mode='gaussian', var=gaussian_var)
    image = np.round(image * 255.).astype(np.uint8)
    iio.imwrite(save_path, image)