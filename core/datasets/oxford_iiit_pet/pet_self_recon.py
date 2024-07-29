#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from .pet_image import PetImageDataset
from utils.typing import TypePathLike
from typing import Union, Tuple, Optional
import torch
import torchvision.transforms.v2 as v2
import numpy.typing as npt
import numpy as np


class PetSelfReconDataset(PetImageDataset):

    def __getitem__(self, index: int):
        data = super().__getitem__(index)

        image = data.pop('image')
        data['source_image'] = image
        data['target_image'] = image
        return data


class TrainingDataset(PetSelfReconDataset):

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
            v2.RandomVerticalFlip(),
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(30, translate=(0.2, 0.2), shear=0.2),
        ])


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


class TestDataset(PetSelfReconDataset):

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