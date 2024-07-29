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
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Any, List, Optional
from ..networks.nets import Segmentor
from ..networks.losses.dice_loss import DiceLoss
import logging
from utils import torch_utils
from utils.typing import TypePathLike
from utils.eval_utils import torch_eval_util
from tqdm import tqdm
import numpy.typing as npt
from utils.typing import TypeNPDTypeFloat
from matplotlib import cm

_logger = logging.getLogger(__name__)


class Training(BaseTrainingModel):

    def __init__(
            self,
            image_channels: int,
            image_size: Union[Tuple[int, int], int],
            n_classes: int,
            class_names: List[str],
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
        self._n_classes = n_classes
        self._class_names = class_names

        valid_backbones = Segmentor.supported_backbones
        if backbone not in valid_backbones:
            raise NotImplementedError(
                f'Unknown backbone {backbone}. '
                f'The supported backbones are {valid_backbones}.'
            )
        self._backbone_name = backbone

        net = Segmentor(
            in_channels=self._image_channels,
            image_size=self._image_size,
            n_classes=self._n_classes,
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

        self._crit_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self._crit_dc = DiceLoss()

        self._optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=learning_rate, weight_decay=1e-4)
        self._grad_scaler = torch.amp.grad_scaler.GradScaler()
        self._optimizers = [self._optimizer]
        self._grad_scalers = [self._grad_scaler]

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
        label = data['label'].to(torch.long).to(self._device)  # (B, H, W)
        pred_logits = self._net(image)
        pred_logits = F.interpolate(
            pred_logits,
            size=self._image_size,
            mode='bilinear',
            align_corners=True,
        )

        loss_ce = self._crit_ce(pred_logits, label)
        pred_prob = torch.softmax(pred_logits, dim=1)
        label = F.one_hot(label, self._n_classes).to(pred_prob.dtype)
        label = torch.einsum('bhwc->bchw', label)
        loss_dc = self._crit_dc(pred_prob, label)

        loss_all = loss_dc + loss_dc
        loss_log = {
            'ce': loss_ce.detach(),
            'dice': loss_dc.detach(),
        }
        return {
            'loss': (loss_all, ),
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

        dice_scores = []
        jaccard_indexs = []
        for data in iterator:
            image = data['image'].to(self._device)
            label = data['label'].to(self._device) # (B, H, W)
            pred_logits = self._net(image)
            pred_logits = F.interpolate(
                pred_logits,
                size=self._image_size,
                mode='bilinear',
                align_corners=True,
            )

            # (B, H, W)
            pred_label = torch.argmax(pred_logits, dim=1, keepdim=False)
            del pred_logits

            pred_label = F.one_hot(pred_label, self._n_classes).to(torch.float32)
            pred_label = torch.einsum('bhwc->bchw', pred_label)

            label = F.one_hot(label, self._n_classes).to(torch.float32)
            label = torch.einsum('bhwc->bchw', label)

            # (B, C)
            dc, jac = torch_eval_util.dice_jaccard(
                x=pred_label, y=label, dim=(2, 3))
            dice_scores.append(dc.cpu())
            jaccard_indexs.append(jac.cpu())
        dice_scores = torch.cat(dice_scores, dim=0)  # (N, C)
        jaccard_indexs = torch.cat(jaccard_indexs, dim=0)  # (N, C)

        eval_results = {
            'mdice': dice_scores.mean(),
            'mjac': jaccard_indexs.mean(),
            'mdice_no_bg': dice_scores[:, 1:].mean(),
            'mjac_no_bg': jaccard_indexs[:, 1:].mean(),
        }

        for i, class_name in enumerate(self._class_names):
            cdice = dice_scores[:, i].mean()
            cjac = jaccard_indexs[:, i].mean()
            eval_results[f'mdice_{class_name}'] = cdice
            eval_results[f'mjac_{class_name}'] = cjac
        return eval_results

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

        images = []
        labels = []
        pred_labels = []
        for data in iterator:
            image = data['image'].to(self._device)  # (B, C, H, W)
            label = data['label'].to(self._device)  # (B, H, W)
            norm_mean = data['norm_mean'].to(self._device)  # (B, C)
            norm_std = data['norm_std'].to(self._device)  # (B, C)
            pred_logits = self._net(image)
            pred_logits = F.interpolate(
                pred_logits,
                size=self._image_size,
                mode='bilinear',
                align_corners=True,
            )
            pred_label = torch.argmax(pred_logits, dim=1, keepdim=False)
            del pred_logits

            norm_mean = norm_mean[..., None, None]  # (B, C, 1, 1)
            norm_std = norm_std[..., None, None]

            image = image * norm_std + norm_mean
            image = image.clamp_(0., 1.)
            image = image.cpu().numpy()
            image = np.einsum('bchw->bhwc', image)

            label = label.to(torch.float32).cpu().numpy()
            pred_label = pred_label.to(torch.float32).cpu().numpy()
            label = label / (self._n_classes - 1)
            pred_label = pred_label / (self._n_classes - 1)
            label = cm.viridis(label)[...,:3]
            pred_label = cm.viridis(pred_label)[..., :3]

            images.append(image)
            labels.append(label)
            pred_labels.append(pred_label)

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)

        res = {
            'images': images,
            'labels': labels,
            'pred_labels': pred_labels,
        }

        resolution = max(self._image_size)
        pad_length = max(round(resolution * 0.02), 1)
        pads = (
            (0, 0),
            (pad_length, pad_length),
            (pad_length, pad_length),
            (0, 0)
        )

        for tag in res:
            image = res[tag]
            image = np.pad(
                image, pads, mode='constant', constant_values=1.)
            res[tag] = image
        return res
