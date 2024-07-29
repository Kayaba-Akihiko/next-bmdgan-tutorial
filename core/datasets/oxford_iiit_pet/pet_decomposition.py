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
import torchvision.transforms.v2 as v2
from .protocol import SampleProtocol
from ..transforms import RandomVerticalFlip, RandomHorizontalFlip
from skimage.util import random_noise

_logger = logging.getLogger(__name__)


@dataclass
class _ImageSample(SampleProtocol):
    image_size: Tuple[int, int]  # (H, W)
    image_path: Path
    add_noise: bool
    keep_original_image: bool = False
    image: Optional[npt.NDArray[np.float32]] = None
    original_image: Optional[npt.NDArray[np.float32]] = None

    def load(self) -> Self:
        image = iio.imread(self.image_path)
        image = image[..., :3]
        image = image.astype(np.float32) / 255.

        if self.keep_original_image:
            self.original_image = image.astype(np.float32, copy=True)

        if self.add_noise:
            image = random_noise(
                image, mode='gaussian', var=0.10)

        image = image_utils.resize(image, self.image_size, order=1)
        self.image = image.astype(np.float32, copy=False)
        return self

@dataclass
class _LabelImageSample(SampleProtocol):
    image_size: Tuple[int, int]  # (H, W)
    image_path: Path

    keep_original_image: bool = False
    image: Optional[npt.NDArray[np.uint8]] = None
    original_image: Optional[npt.NDArray[np.uint8]] = None

    def load(self) -> Self:

        # 1, 2, 3 (pet, background, boarder)
        label = iio.imread(self.image_path)
        # 0, 1, 2 (bg, pet, boarder)
        re_label = np.zeros_like(label)
        re_label[label == 1] = 1
        re_label[label == 3] = 2
        label = re_label
        del re_label

        if self.keep_original_image:
            self.original_image = label.astype(np.uint8, copy=True)

        label = image_utils.resize(label, self.image_size, order=0)
        self.image = label.astype(np.uint8, copy=False)
        return self


@dataclass
class Sample(SampleProtocol):
    sample_id: str
    clean_image_sample: _ImageSample
    noised_image_sample: _ImageSample
    label_sample: _LabelImageSample

    def load(self) -> Self:
        if self.clean_image_sample.add_noise:
            raise RuntimeError('This is supposed to be a clean image.')
        self.clean_image_sample.load()
        self.noised_image_sample.load()
        self.label_sample.load()
        return self

class DecompositionBaseDataset(BaseDataset):

    def __init__(
            self,
            data_root: TypePathLike,
            load_size: Union[Tuple[int, int], int],
            image_size: Union[Tuple[int, int], int],
            split: Literal['train', 'test'],
            mode: Literal['pet', 'pet_and_boarder'],  # (stage two, stage 1)
            preload=False,
            n_workers: Optional[int] = None,
            use_online_noise: bool = False,
            return_original_image: bool = False,  # For fair evaluation
            allow_samples: Union[Literal['all'], List[str]]='all',
    ):
        super().__init__(data_root, n_workers)

        self._split = split
        self._load_size = container_utils.to_2tuple(load_size)
        self._image_size = container_utils.to_2tuple(image_size)
        self._preload = preload
        self._mode = mode
        self._return_original_image = return_original_image

        if self._split == 'train':
            list_file_path = self._anns_folder / 'trainval.txt'
        else:
            list_file_path = self._anns_folder / 'test.txt'

        clean_image_folder = self._images_folder
        if use_online_noise:
            noised_image_folder = self._images_folder
            add_noise = True
        else:
            noised_image_folder = self._noised_images_folder
            add_noise = False
        if allow_samples != 'all':
            _logger.info(f'Allow samples: {allow_samples}.')
            allow_samples = set(allow_samples)
        pool = []
        with list_file_path.open('r') as f:
            for i, line in enumerate(f):
                line_comp = line.strip().split()
                image_name, class_id, specie_id, breed_id = line_comp
                if allow_samples != 'all':
                    if image_name not in allow_samples:
                        continue
                clean_image_path = clean_image_folder / f'{image_name}.jpg'
                noised_image_path = noised_image_folder / f'{image_name}.jpg'
                label_path = self._trimaps_folder / f'{image_name}.png'

                clean_image_sample = _ImageSample(
                    self._load_size,
                    clean_image_path,
                    add_noise=False,
                    keep_original_image=self._return_original_image,
                )
                noised_image_sample = _ImageSample(
                    self._load_size,
                    noised_image_path,
                    add_noise=add_noise,
                    keep_original_image=self._return_original_image,
                )
                label_sample = _LabelImageSample(
                    self._load_size,
                    label_path,
                    keep_original_image=self._return_original_image,
                )

                sample = Sample(
                    image_name,
                    clean_image_sample,
                    noised_image_sample,
                    label_sample,
                )
                pool.append(sample)
        self._pool = pool

        if self._preload:
            self._preload_pool(desc=f'Preloading data ({self._split})')

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._pool[index]
        if self._preload:
            clean_image = sample.clean_image_sample.image.astype(
                np.float32, copy=True)
            noised_image= sample.noised_image_sample.image.astype(
                np.float32, copy=True)
            label = sample.label_sample.image.astype(
                np.uint8, copy=True)
        else:
            sample = copy.deepcopy(sample)
            sample.load()
            clean_image = sample.clean_image_sample.image.astype(
                np.float32, copy=False)
            noised_image = sample.noised_image_sample.image.astype(
                np.float32, copy=False)
            label = sample.label_sample.image.astype(
                np.uint8, copy=False)

        clean_image, noised_image, label = self._augment_image(
            clean_image, noised_image, label
        )

        clean_image = image_utils.resize(
            clean_image, self._image_size, order=1)
        noised_image = image_utils.resize(
            noised_image, self._image_size, order=1)
        label = image_utils.resize(
            label, self._image_size, order=0)  # (H, W)

        if self._mode == 'pet':
            mask = label == 1
        elif self._mode == 'pet_and_boarder':
            mask = np.logical_or(label == 1, label == 2)
        else:
            raise NotImplementedError(self._mode)
        clean_image[np.logical_not(mask)] = 0

        # normalize
        clean_image = (clean_image - self._norm_mean) / self._norm_std
        noised_image = (noised_image - self._norm_mean) / self._norm_std

        # (H, W, C) -> (C, H, W)
        clean_image = np.einsum('ijk->kij', clean_image)
        noised_image = np.einsum('ijk->kij', noised_image)

        with torch.no_grad():
            clean_image = torch.from_numpy(clean_image)
            noised_image = torch.from_numpy(noised_image)
            norm_mean = torch.from_numpy(self._norm_mean)
            norm_std = torch.from_numpy(self._norm_std)
        res = {
            'sample_id': sample.sample_id,
            'source_image': noised_image,
            'target_image': clean_image,
            'norm_mean': norm_mean,
            'norm_std': norm_std,
        }

        if self._return_original_image:
            # (H, W, C)
            clean_image = sample.clean_image_sample.original_image
            noised_image = sample.noised_image_sample.original_image
            label = sample.label_sample.original_image

            if self._mode == 'pet':
                mask = label == 1
            elif self._mode == 'pet_and_boarder':
                mask = np.logical_or(label == 1, label == 2)
            else:
                raise NotImplementedError(self._mode)

            clean_image[np.logical_not(mask)] = 0  #
            clean_image = np.einsum('ijk->kij', clean_image)
            noised_image = np.einsum('ijk->kij', noised_image)

            with torch.no_grad():
                clean_image = torch.from_numpy(clean_image)
                noised_image = torch.from_numpy(noised_image)

            res['original_target_image'] = clean_image
            res['original_source_image'] = noised_image

        return res

    @abstractmethod
    def _augment_image(
            self,
            clean_image: npt.NDArray[np.float32],
            noised_image: npt.NDArray[np.float32],
            label: npt.NDArray[np.uint8]
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint8]
    ]:
        raise NotImplementedError


class TrainingDataset(DecompositionBaseDataset):

    def __init__(
            self,
            data_root: TypePathLike,
            load_size: Union[Tuple[int, int], int],
            image_size: Union[Tuple[int, int], int],
            mode: Literal['pet', 'pet_and_boarder'],  # (stage two, stage 1)
            preload=False,
            n_workers: Optional[int] = None,
            use_online_noise=True,
    ):
        super().__init__(
            data_root=data_root,
            load_size=load_size,
            image_size=image_size,
            split='train',
            mode=mode,
            preload=preload,
            n_workers=n_workers,
            use_online_noise=use_online_noise,
        )

        self._transform = v2.Compose([
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
        ])

    @torch.no_grad()
    def _augment_image(
            self,
            clean_image: npt.NDArray[np.float32],
            noised_image: npt.NDArray[np.float32],
            label: npt.NDArray[np.uint8]
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint8]
    ]:
        # (H, W, C) -> (C, H, W)
        clean_image = np.einsum(
            'ijk->kij', clean_image)
        noised_image = np.einsum(
            'ijk->kij', noised_image)
        label = np.expand_dims(label, axis=0)  # (1, H, W)

        clean_image = torch.from_numpy(clean_image)
        noised_image = torch.from_numpy(noised_image)
        label = torch.from_numpy(label)
        clean_image, noised_image, label = self._transform(
            clean_image, noised_image, label)
        clean_image = clean_image.numpy()
        noised_image = noised_image.numpy()
        label = label.numpy()
        # (C, H, W) -> (H, W, C)
        clean_image = np.einsum(
            'ijk->jki', clean_image)
        noised_image = np.einsum(
            'ijk->jki', noised_image)
        label = label[0]
        return clean_image, noised_image, label


class TestDataset(DecompositionBaseDataset):

    def __init__(
            self,
            data_root: TypePathLike,
            image_size: Union[Tuple[int, int], int],
            mode: Literal['pet', 'pet_and_boarder'],  # (stage two, stage 1)
            preload=False,
            n_workers: Optional[int] = None,
            use_online_noise=False,
            return_original_image: bool = False,  # True for make fair evaluation
            allow_samples: Union[Literal['all'], List[str]] = 'all',
    ):
        super().__init__(
            data_root=data_root,
            load_size=image_size,
            image_size=image_size,
            split='test',
            mode=mode,
            preload=preload,
            n_workers=n_workers,
            use_online_noise=use_online_noise,
            return_original_image=return_original_image,
            allow_samples=allow_samples,
        )

    def _augment_image(
            self,
            clean_image: npt.NDArray[np.float32],
            noised_image: npt.NDArray[np.float32],
            label: npt.NDArray[np.uint8],
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint8]
    ]:
        return clean_image, noised_image, label
