#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import copy

from .base_dataset import BaseDataset
import logging
from utils.typing import TypePathLike
from utils import image_utils, container_utils
from typing import Literal, Optional, Tuple, Self, Union, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import numpy.typing as npt
import numpy as np
import imageio.v3 as iio
from abc import abstractmethod
import torch
from .protocol import SampleProtocol
import torchvision.transforms.v2 as v2
from ..transforms import RandomVerticalFlip, RandomHorizontalFlip
from skimage.util import random_noise

@dataclass
class _ImageSample(SampleProtocol):
    image_size: Tuple[int, int]
    image_path: Path
    label_path: Path
    add_noise: bool

    image: Optional[npt.NDArray[np.float32]] = None
    label: Optional[npt.NDArray[np.uint8]] = None

    def load(self) -> Self:
        image = iio.imread(self.image_path)
        image = image[..., :3]
        image = image.astype(np.float32) / 255.

        if self.add_noise:
            image = random_noise(
                image, mode='gaussian', var=0.10)

        # 1, 2, 3 (pet, background, boarder)
        label = iio.imread(self.label_path)
        # 0, 1, 2 (bg, pet, boarder)
        re_label = np.zeros_like(label)
        re_label[label == 1] = 1
        re_label[label == 3] = 2
        label = re_label
        del re_label

        image = image_utils.resize(image, self.image_size, order=1)
        label = image_utils.resize(label, self.image_size, order=0)
        self.image = image.astype(np.float32, copy=False)
        self.label = label.astype(np.uint8, copy=False)
        return self


class SegmentationBaseDataset(BaseDataset):

    def __init__(
            self,
            data_root: TypePathLike,
            load_size: Union[Tuple[int, int], int],
            image_size: Union[Tuple[int, int], int],
            split: Literal['train', 'test'],
            preload=False,
            n_workers: Optional[int] = None,
            use_noised_images: bool = True,
            use_online_noise: bool = False,
    ):
        super().__init__(data_root, n_workers)

        self._split = split
        self._load_size = container_utils.to_2tuple(load_size)
        self._image_size = container_utils.to_2tuple(image_size)
        self._preload = preload

        if self._split == 'train':
            list_file_path = self._anns_folder / 'trainval.txt'
        else:
            list_file_path = self._anns_folder / 'test.txt'

        add_noise = False
        image_folder = self._images_folder
        if use_noised_images:
            if use_online_noise:
                add_noise = True
            else:
                # deterministic noise
                image_folder = self._noised_images_folder

        pool = []
        with list_file_path.open('r') as f:
            for i, line in enumerate(f):
                line_comp = line.strip().split()
                image_name, class_id, specie_id, breed_id = line_comp
                image_path = image_folder / f'{image_name}.jpg'
                label_path = self._trimaps_folder / f'{image_name}.png'
                sample = _ImageSample(
                    self._load_size,
                    image_path,
                    label_path,
                    add_noise=add_noise,
                )
                pool.append(sample)

        self._pool = pool

        if self._preload:
            self._preload_pool(desc=f'Preloading data ({self._split})')

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._pool[index]
        if self._preload:
            image = sample.image.astype(np.float32, copy=True)
            label = sample.label.astype(np.uint8, copy=True)
        else:
            sample = copy.deepcopy(sample)
            sample.load()
            image = sample.image.astype(np.float32, copy=False)
            label = sample.label.astype(np.uint8, copy=False)

        image, label = self._augment_image(image, label)
        image = image_utils.resize(image, self._image_size, order=1)
        label = image_utils.resize(label, self._image_size, order=0)
        # normalize
        image = (image - self._norm_mean) / self._norm_std  # (H, W, C)
        # (H, W, C) -> (C, H, W)
        image = np.einsum('ijk->kij', image)
        with torch.no_grad():
            image = torch.from_numpy(image)
            label = torch.from_numpy(label).to(torch.long)
            norm_mean = torch.from_numpy(self._norm_mean)
            norm_std = torch.from_numpy(self._norm_std)
        return {
            'image': image,
            'label': label,
            'norm_mean': norm_mean,
            'norm_std': norm_std,
            }

    @abstractmethod
    def _augment_image(
            self,
            image: npt.NDArray[np.float32],
            label: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        raise NotImplementedError


class TrainingDataset(SegmentationBaseDataset):
    def __init__(
            self,
            data_root: TypePathLike,
            load_size: Union[Tuple[int, int], int],
            image_size: Union[Tuple[int, int], int],
            preload=False,
            n_workers: Optional[int] = None,
            use_noised_images=True,
            use_online_noise=True,
    ):
        super().__init__(
            data_root=data_root,
            load_size=load_size,
            image_size=image_size,
            split='train',
            preload=preload,
            n_workers=n_workers,
            use_noised_images=use_noised_images,
            use_online_noise=use_online_noise,
        )

        self._transform = v2.Compose([
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
        ])

    @torch.no_grad()
    def _augment_image(
            self,
            image: npt.NDArray[np.float32],
            label: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        # (H, W, C) -> (C, H, W)
        image = np.einsum('ijk->kij', image)
        label = np.expand_dims(label, axis=0)  # (1, H, W)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        image, label = self._transform(image, label)
        image = image.numpy()
        label = label.numpy()
        # (C, H, W) -> (H, W, C)
        image = np.einsum('ijk->jki', image)
        label = label[0]
        return image, label


class TestDataset(SegmentationBaseDataset):

    def __init__(
            self,
            data_root: TypePathLike,
            image_size: Union[Tuple[int, int], int],
            preload=False,
            n_workers: Optional[int] = None,
            use_noised_images=True,
            use_online_noise=False,
    ):
        super().__init__(
            data_root=data_root,
            load_size=image_size,
            image_size=image_size,
            split='test',
            preload=preload,
            n_workers=n_workers,
            use_noised_images=use_noised_images,
            use_online_noise=use_online_noise,
        )

    def _augment_image(
            self,
            image: npt.NDArray[np.float32],
            label: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        return image, label


