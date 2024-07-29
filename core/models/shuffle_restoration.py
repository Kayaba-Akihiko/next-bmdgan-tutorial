#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from .base_model import BaseTrainingModel
import torch
from torch.utils.data import DataLoader
from typing import Union, Tuple, Dict, Any, List, Optional
import logging
from utils import torch_utils
from utils.typing import TypePathLike
from tqdm import tqdm
import numpy.typing as npt
from utils.typing import TypeNPDTypeFloat
from einops import rearrange
from ..networks.nets import Classifier
import numpy as np
import logging
from .masked_autoencoder import patchify_image, unpatchify
import math

_logger = logging.getLogger(__name__)

class Training(BaseTrainingModel):
    def __init__(
            self,
            image_channels: int,
            image_size: Union[Tuple[int, int], int],
            backbone: str,
            n_patches = 64,
            learning_rate: float = 2e-4,
            device=torch.device('cpu'),
            pretrain_backbone_load_path: Optional[TypePathLike] = None,
            pretrain_backbone_strict_load=True,
    ):
        super().__init__(
            image_channels=image_channels,
            image_size=image_size,
            device=device,
        )

        h, w = self._image_size

        if h != w:
            raise NotImplementedError(self._image_size)
        self._n_patches_y = int(math.sqrt(n_patches))
        self._n_patches_x = self._n_patches_y
        self._n_patches = self._n_patches_x * self._n_patches_y
        self._patch_size_y = h // self._n_patches_y
        if h % self._n_patches_y != 0:
            raise ValueError(
                f'Patch size must be divisible by {self._n_patches_y}')
        self._patch_size_x = w // self._n_patches_x
        if w % self._n_patches_x != 0:
            raise ValueError(
                f'Patch size must be divisible by {self._n_patches_x}')

        valid_backbones = Classifier.supported_backbones
        if backbone not in valid_backbones:
            raise NotImplementedError(
                f'Unknown backbone {backbone}. '
                f'The supported backbones are {valid_backbones}.'
            )
        self._backbone_name = backbone

        net = Classifier(
            in_channels=self._image_channels,
            image_size=self._image_size,
            n_classes=self._n_patches ** 2,
            backbone_cfg={'arch': self._backbone_name},
            pretrain_backbone_load_path=pretrain_backbone_load_path,
            pretrain_backbone_strict_load=pretrain_backbone_strict_load,
        ).to(self._device)
        self._net = net
        self._register_network('net', self._net)
        n_train, n_tix, n_total = torch_utils.count_parameters(self._net)
        _logger.info(
            f"{self._net.__class__} has "
            f"{n_total * 1.e-6:.2f} M params "
            f"({n_train * 1.e-6:.2f} trainable)."
        )

        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        grad_scaler = torch.amp.grad_scaler.GradScaler()

        self._optimizers = [optimizer]
        self._grad_scalers = [grad_scaler]

        self._crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    @property
    def optimizers(self) -> List[torch.optim.Optimizer]:
        return self._optimizers

    def _compute_loss(
            self,
            data: Dict[str, Any],
            epoch: int,
    ) -> Dict[
        str,
        Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]
    ]:
        with torch.no_grad():
            image = data['image'].to(self._device)
            # (B, L, D)
            x = patchify_image(
                image, self._n_patches_y, self._n_patches_x)
            B, L, D = x.shape
            shuffle_res = _shuffle_patches(x)
            del x
            shuffled_x = shuffle_res['shuffled_x']
            ids_restore = shuffle_res['ids_restore']  # (B, L)
            del shuffle_res
            shuffled_image = unpatchify(
                shuffled_x,
                self._image_size,
                self._n_patches_y,
                self._n_patches_x,
                self._image_channels
            )
            del shuffled_x

        pred_logits = self._net(shuffled_image)  # (B, L * L)
        pred_logits = pred_logits.view(B, L, L).contiguous()
        loss = self._crit(pred_logits, ids_restore)
        loss_log = {
            'ce': loss.detach()
        }
        return {
            'loss': (loss,),
            'log': loss_log
        }

    @torch.no_grad()
    def evaluate_epoch(
            self, test_data_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        n_iter = len(test_data_loader)
        iterator = tqdm(
            test_data_loader,
            desc='Evaluating',
            total=n_iter,
            mininterval=30, maxinterval=60,
        )

        accuracies = []
        for data in iterator:
            image = data['image'].to(self._device)
            # (B, L, D)
            x = patchify_image(
                image, self._n_patches_y, self._n_patches_x)
            B, L, D = x.shape
            shuffle_res = _shuffle_patches(x)
            del x
            shuffled_x = shuffle_res['shuffled_x']
            ids_restore = shuffle_res['ids_restore']  # (B, L)
            del shuffle_res
            shuffled_image = unpatchify(
                shuffled_x,
                self._image_size,
                self._n_patches_y,
                self._n_patches_x,
                self._image_channels
            )
            pred_logits = self._net(shuffled_image)  # (B, L * L)
            del shuffled_image
            pred_logits = pred_logits.view(B, L, L).contiguous()
            pred_labels = pred_logits.argmax(dim=-1)  # (B, L)
            del pred_logits

            # (B, L)
            accuracy = (pred_labels == ids_restore).to(torch.float32)
            accuracy = torch.sum(accuracy, dim=-1) / L  # (B,)
            accuracies.append(accuracy.cpu())
        accuracies = torch.cat(accuracies, dim=0)  # (N,
        macc = accuracies.mean()

        return {
            'accuracy': macc,
        }

    @torch.no_grad()
    def visualize_epoch(
            self, visualization_data_loader: DataLoader
    ) -> Dict[str, Union[torch.Tensor, npt.NDArray[TypeNPDTypeFloat]]]:

        n_iter = len(visualization_data_loader)
        iterator = tqdm(
            visualization_data_loader,
            desc='Visualizing',
            total=n_iter,
            mininterval=30, maxinterval=60,
        )

        image_collections = {
            'image': [],
            'shuffled': [],
            'restored': [],
        }
        for data in iterator:
            image = data['image'].to(self._device)
            norm_mean = data['norm_mean'].to(self._device)  # (B, C)
            norm_std = data['norm_std'].to(self._device)  # (B, C)
            # (B, L, D)
            x = patchify_image(
                image, self._n_patches_y, self._n_patches_x)
            B, L, D = x.shape
            shuffle_res = _shuffle_patches(x)
            del x
            shuffled_x = shuffle_res['shuffled_x']
            ids_restore = shuffle_res['ids_restore']  # (B, L)
            del shuffle_res
            shuffled_image = unpatchify(
                shuffled_x,
                self._image_size,
                self._n_patches_y,
                self._n_patches_x,
                self._image_channels
            )
            pred_logits = self._net(shuffled_image)  # (B, L * L)
            pred_logits = pred_logits.view(B, L, L).contiguous()
            pred_labels = pred_logits.argmax(dim=-1)  # (B, L)
            del pred_logits

            restored_x = torch.gather(
                shuffled_x,
                dim=1,
                index=pred_labels.unsqueeze(-1).repeat(1, 1, D),
            )

            restored_image = unpatchify(
                restored_x,
                self._image_size,
                self._n_patches_y,
                self._n_patches_x,
                self._image_channels
            )

            norm_mean = norm_mean[..., None, None]  # (B, C, 1, 1)
            norm_std = norm_std[..., None, None]

            image = image * norm_std + norm_mean
            image = image.clamp_(0., 1.)
            shuffled_image = shuffled_image * norm_std + norm_mean
            shuffled_image = shuffled_image.clamp_(0., 1.)
            restored_image = restored_image * norm_std + norm_mean
            restored_image = restored_image.clamp_(0., 1.)
            image_collections['image'].append(image.cpu())
            image_collections['shuffled'].append(shuffled_image.cpu())
            image_collections['restored'].append(restored_image.cpu())

        resolution = max(self._image_size)
        pad_length = max(round(resolution * 0.02), 1)
        pads = (
            (0, 0),
            (pad_length, pad_length),
            (pad_length, pad_length),
            (0, 0)
        )
        for tag in image_collections:
            # (N, C, H, W)
            images = torch.cat(image_collections[tag], dim=0).numpy()
            # (N, H, W, C)
            images = np.einsum('nchw->nhwc', images)
            images = np.pad(
                images, pads, mode='constant', constant_values=1.)
            image_collections[tag] = images
        return image_collections


def _shuffle_patches(x: torch.Tensor):
    B, L, D = x.shape
    noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    shuffled_x = torch.gather(
        x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
    return {
        'shuffled_x': shuffled_x,
        'ids_shuffle': ids_shuffle,
        'ids_restore': ids_restore,
    }
