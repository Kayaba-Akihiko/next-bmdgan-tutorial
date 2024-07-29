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
import xml.etree.ElementTree as ET
from torchvision import tv_tensors

_logger = logging.getLogger(__name__)

@dataclass
class _ImageSample(SampleProtocol):
    class_id: int
    specie_id: int
    breed_id: int
    image_size: Tuple[int, int]  # (H, W)
    image_path: Path
    bbox_path: Path
    add_noise: bool

    image: Optional[npt.NDArray[np.float32]] = None
    bbox: Optional[npt.NDArray[np.float32]] = None

    def load(self) -> Self:
        image = iio.imread(self.image_path)
        image = image[..., :3]
        image = image.astype(np.float32) / 255.
        if self.add_noise:
            image = random_noise(
                image, mode='gaussian', var=0.10)
        restore_size = image.shape[:2]  # (H, W)
        image = image_utils.resize(image, self.image_size, order=1)
        self.image = image.astype(np.float32, copy=False)

        # xmin, ymin, xmax, ymax
        bbox = self._read_bb(self.bbox_path)[0]
        bbox = np.asarray(bbox, dtype=np.float32)
        self.bbox = bbox.astype(np.float32, copy=False)

        return self

    @staticmethod
    def _read_bb(xml_file: Path) -> np.ndarray:

        tree = ET.parse(xml_file)
        root = tree.getroot()

        list_with_all_boxes = []

        w = float(root.find('size/width').text)
        h = float(root.find('size/height').text)
        for boxes in root.iter('object'):
            ymin = float(boxes.find("bndbox/ymin").text) - 1
            xmin = float(boxes.find("bndbox/xmin").text) - 1
            ymax = float(boxes.find("bndbox/ymax").text) - 1
            xmax = float(boxes.find("bndbox/xmax").text) - 1

            list_with_single_boxes = [
                xmin / w,
                ymin / h,
                xmax / w,
                ymax / h,
            ]
            list_with_all_boxes.append(list_with_single_boxes)

        return np.asarray(list_with_all_boxes, dtype=np.float64)


class SingleBoundingBoxBaseDataset(BaseDataset):

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
                bbox_path = self._xmls_folder / f'{image_name}.xml'
                if not bbox_path.exists():
                    continue
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
                    bbox_path,
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
            bbox = sample.bbox.astype(np.float32, copy=True)
        else:
            sample = copy.deepcopy(sample)
            sample.load()
            image = sample.image.astype(np.float32, copy=False)
            bbox = sample.bbox.astype(np.float32, copy=False)

        image, bbox = self._augment_image(image, bbox)

        H, W, _ = image.shape
        image = image_utils.resize(image, self._image_size, order=1)

        # normalize
        image = (image - self._norm_mean) / self._norm_std  # (H, W, C)
        image = np.einsum('ijk->kij', image)
        with torch.no_grad():
            image = torch.from_numpy(image)
            bbox = torch.tensor(bbox, dtype=torch.float32)
            norm_mean = torch.from_numpy(self._norm_mean)
            norm_std = torch.from_numpy(self._norm_std)
        return {
            'image': image,
            'bbox': bbox,
            'norm_mean': norm_mean,
            'norm_std': norm_std,
        }

    @abstractmethod
    def _augment_image(
            self,
            image: npt.NDArray[np.float32],
            bbox: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        raise NotImplementedError


class TrainingDataset(SingleBoundingBoxBaseDataset):

    def __init__(
            self,
            data_root: TypePathLike,
            load_size: Union[Tuple[int, int], int],
            image_size: Union[Tuple[int, int], int],
            preload=False,
            n_workers: Optional[int] = None,
            use_noised_images=True,
            use_online_noise=True,
            with_aug=True,
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
            v2.RandomVerticalFlip(),
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(60, translate=(0.3, 0.3), shear=0.3),
        ])
        self._with_aug = with_aug

    @torch.no_grad()
    def _augment_image(
            self,
            image: npt.NDArray[np.float32],
            bbox: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        if not self._with_aug:
            return image, bbox
        # (H, W, C) -> (C, H, W)
        H, W, C = image.shape
        image = np.einsum('ijk->kij', image)
        image = torch.from_numpy(image)
        image = tv_tensors.Image(image)

        # XYXY
        sizes = np.asarray([W, H, W, H], dtype=np.float64)
        bbox = np.multiply(bbox, sizes)  # to integer
        # bbox = np.round(bbox)
        bbox = tv_tensors.BoundingBoxes(
            bbox.reshape(1, 4),
            format="XYXY",
            canvas_size=(H, W),
        )

        image, bbox = self._transform(image, bbox)

        image = torch.as_tensor(image)
        bbox = torch.as_tensor(bbox) # (1, 4)

        image = image.numpy()
        bbox = bbox[0].numpy().astype(np.float64, copy=False)
        bbox = np.divide(bbox, sizes) # to float

        bbox = np.asarray(bbox, dtype=np.float64)
        # (C, H, W) -> (H, W, C)
        image = np.einsum('ijk->jki', image)
        return image, bbox


class TestDataset(SingleBoundingBoxBaseDataset):

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
            bbox: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        return image, bbox