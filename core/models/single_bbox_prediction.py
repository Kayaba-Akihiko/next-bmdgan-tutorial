#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.
import numpy as np

from .base_model import BaseTrainingModel
import torch
from torch.utils.data import DataLoader
from typing import Union, Tuple, Dict, Any, List, Optional
from ..networks.nets import Classifier
import logging
from utils import torch_utils
from tqdm import tqdm
import numpy.typing as npt
from utils.typing import TypeNPDTypeFloat, TypePathLike
from torchvision.ops import generalized_box_iou_loss, generalized_box_iou
import cv2

_logger = logging.getLogger(__name__)


class Training(BaseTrainingModel):

    def __init__(
            self,
            image_channels: int,
            image_size: Union[Tuple[int, int], int],
            backbone: str,
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
            n_classes=4,
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

        self._crit = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=learning_rate, weight_decay=1e-4)
        grad_scaler = torch.amp.grad_scaler.GradScaler()
        self._optimizers = [optimizer]
        self._grad_scalers = [grad_scaler]

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
        image = data['image'].to(self._device)
        bbox = data['bbox'].to(self._device)  # (B, 4)
        pred_bbox = self._net(image)
        reg_loss = self._crit(pred_bbox, bbox)
        pred_bbox = torch.relu(pred_bbox)  # (0, +)
        pred_bbox = -pred_bbox  # (-, 0)
        pred_bbox = pred_bbox + 1 # (-, 1
        pred_bbox = torch.relu(pred_bbox)  # (0, 1)
        pred_bbox = 1 - pred_bbox
        iou_loss = torch.abs(
            generalized_box_iou_loss(
                pred_bbox, bbox, reduction='mean')
        )

        loss_log: Dict[str, torch.Tensor] = {
            'reg': reg_loss.detach(),
            'iou': iou_loss.detach(),
        }

        return {
            'loss': (reg_loss + iou_loss,),
            'log': loss_log,
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

        total_iou = torch.tensor(
            [0], dtype=torch.float32, device=self._device)
        total_count = 0
        for data in iterator:
            image = data['image'].to(self._device)
            bbox = data['bbox'].to(self._device)  # (B, 4)
            pred_bbox = self._net(image)
            B = len(image)
            for i in range(B):
                iou = generalized_box_iou(
                    pred_bbox[i: i + 1], bbox[i: i + 1])
                iou = iou.view(1)
                total_iou += iou
                total_count += 1
        miou = total_iou / total_count
        return {
            'miou': miou,
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

        image_collection = {
            'image': [],
            'pred_bbox': [],
            'gt_bbox': [],
        }

        for data in iterator:
            batch_image = data['image'].to(self._device)
            batch_bbox = data['bbox'].to(self._device)  # (B, 4)
            batch_pred_bbox = self._net(batch_image)
            batch_pred_bbox.clamp_(0., 1.)

            norm_mean = data['norm_mean'].to(self._device)  # (B, C)
            norm_std = data['norm_std'].to(self._device)  # (B, C)

            norm_mean = norm_mean[..., None, None]  # (B, C, 1, 1)
            norm_std = norm_std[..., None, None]

            batch_image = batch_image * norm_std + norm_mean
            batch_image = batch_image.clamp_(0., 1.)

            batch_image = batch_image.cpu().numpy()
            batch_bbox = batch_bbox.cpu().numpy()
            batch_pred_bbox = batch_pred_bbox.cpu().numpy()
            batch_image = np.einsum('bchw->bhwc', batch_image)

            B = len(batch_image)
            for i in range(B):
                image = batch_image[i]  # (H, W, C)
                bbox = batch_bbox[i]
                pred_box = batch_pred_bbox[i]
                gt_labeled_image = _draw_bbox_on_image(
                    image.copy(), bbox, thickness=2)
                pred_labeled_image = _draw_bbox_on_image(
                    image.copy(), pred_box, thickness=2)
                image_collection['image'].append(image)
                image_collection['gt_bbox'].append(gt_labeled_image)
                image_collection['pred_bbox'].append(pred_labeled_image)

        resolution = max(self._image_size)
        pad_length = max(round(resolution * 0.02), 1)
        pads = (
            (0, 0),
            (pad_length, pad_length),
            (pad_length, pad_length),
            (0, 0)
        )
        for tag in image_collection:
            images = np.stack(image_collection[tag])  # (N, H, W, C)
            images = np.pad(
                images, pads, mode='constant', constant_values=1.)
            image_collection[tag] = images
        return image_collection


def _draw_bbox_on_image(
        image, bbox: npt.NDArray, color=(10, 255, 0), thickness=1):
    H, W, _ = image.shape
    bbox = bbox.tolist()
    res = np.round(image * 255.).astype(np.uint8, copy=True)
    pt1 = (round(bbox[0] * W), round(bbox[1] * H))
    pt2 = (round(bbox[2] * W), round(bbox[3] * H))
    res = cv2.rectangle(res, pt1, pt2, color, thickness)
    res = res.astype(np.float32) / 255.
    return res