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
from typing import Literal, Optional, Tuple, Self, Union, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy.typing as npt
import numpy as np
import imageio.v3 as iio
from abc import abstractmethod
import torch
import torchvision.transforms.v2 as v2
from .protocol import SampleProtocol
from skimage.util import random_noise
import skimage.transform as transform

_logger = logging.getLogger(__name__)


@dataclass
class _ImageSample(SampleProtocol):
    class_id: int
    specie_id: int
    breed_id: int
    image_size: Tuple[int, int]  # (H, W)
    image_path: Path
    add_noise: bool

    image: Optional[npt.NDArray[np.float32]] = None

    def load(self) -> Self:
        image = iio.imread(self.image_path)
        image = image[..., :3]
        image = image.astype(np.float32) / 255.
        if self.add_noise:
            image = random_noise(
                image, mode='gaussian', var=0.10)
        image = image_utils.resize(image, self.image_size, order=1)
        self.image = image.astype(np.float32, copy=False)
        return self



class PetImageRotationBaseDataset(BaseDataset):
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

        self._rotation_angles = [0, 90, 180, 270]

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
                class_id = int(class_id)
                specie_id = int(specie_id)
                breed_id = int(breed_id)
                image_path = image_folder / f'{image_name}.jpg'
                sample = _ImageSample(
                    class_id - 1,
                    specie_id - 1,
                    breed_id - 1,
                    self._load_size,
                    image_path,
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
        else:
            sample = copy.deepcopy(sample)
            sample.load()
            image = sample.image.astype(np.float32, copy=False)

        image = self._augment_image(image)

        label = np.random.randint(len(self._rotation_angles))
        angle = self._rotation_angles[label]
        image = transform.rotate(image, angle)

        image = image_utils.resize(image, self._image_size, order=1)
        # normalize
        image = (image - self._norm_mean) / self._norm_std  # (H, W, C)
        # (H, W, C) -> (C, H, W)
        image = np.einsum('ijk->kij', image)
        with torch.no_grad():
            image = torch.from_numpy(image)
            label = torch.tensor(label, dtype=torch.long)
        return {
            'image': image,
            'label': label,
        }

    @abstractmethod
    def _augment_image(
            self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        raise NotImplementedError


class TrainingDataset(PetImageRotationBaseDataset):

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

        self._transform = v2.RandomResizedCrop(
            size=self._load_size,
            scale=(0.8, 1.0),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )

    @torch.no_grad()
    def _augment_image(
            self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # (H, W, C) -> (C, H, W)
        image = np.einsum('ijk->kij', image)
        image = torch.from_numpy(image)
        image = self._transform(image)
        image = image.numpy()
        # (C, H, W) -> (H, W, C)
        image = np.einsum('ijk->jki', image)
        return image


class TestDataset(PetImageRotationBaseDataset):

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
            self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return image