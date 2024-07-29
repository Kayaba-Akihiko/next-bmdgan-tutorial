#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.



from .base_model import BaseTrainingModel
import torch
from torch.utils.data import DataLoader
from typing import Union, Tuple, Dict, Any, List, Optional, Literal
from ..networks.nets import Segmentor
import logging
from utils import torch_utils
from utils.typing import TypePathLike
from tqdm import tqdm
import numpy.typing as npt
from utils.typing import TypeNPDTypeFloat

from ..networks.nets import ContrastiveLearningEncoder
from ..networks.losses.sup_con_loss import SupConLoss

_logger = logging.getLogger(__name__)


class Training(BaseTrainingModel):

    def __init__(
            self,
            image_channels: int,
            image_size: Union[Tuple[int, int], int],
            backbone: str,
            projection_dim = 128,
            use_mlp: bool = True,
            contrast_mode: Literal[
                'unsupervised', 'supervised'] = 'unsupervised',
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
        self._contrast_mode = contrast_mode

        valid_backbones = Segmentor.supported_backbones
        if backbone not in valid_backbones:
            raise NotImplementedError(
                f'Unknown backbone {backbone}. '
                f'The supported backbones are {valid_backbones}.'
            )
        self._backbone_name = backbone
        net = ContrastiveLearningEncoder(
            in_channels=self._image_channels,
            image_size=self._image_size,
            projection_dim=projection_dim,
            backbone_cfg={'arch': self._backbone_name},
            use_mlp=use_mlp,
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

        self._criterion = SupConLoss().to(self._device)
        self._register_network('criterion', self._criterion)

        self._optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
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
        # B, C, H, W)
        image_query = data['image_query']
        image_key = data['image_key']
        images = torch.cat(
            [image_query, image_key], dim=0).to(self._device)

        # (2*B, C, H, W)
        feats = self._net(images)
        q, k = feats.chunk(2, dim=0)


        if self._contrast_mode == 'supervised':
            label = data['label'].to(self._device)
        elif self._contrast_mode == 'unsupervised':
            label = None
        else:
            raise NotImplementedError()
        loss = self._criterion(q, k, label)
        loss_log = {
            'con_loss': loss.detach()
        }
        return {
            'loss': (loss,),
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

        # This is just a rough evaluation.
        # The scrip below tries to evaluate the accuracy of finding a positive key from the batch.
        accurate_count = torch.tensor(
            0, dtype=torch.float32, device=self._device)
        total_count = torch.tensor(
            0, dtype=torch.float32, device=self._device)

        for data in iterator:
            # B, C, H, W)
            image_query = data['image_query']
            image_key = data['image_key']
            B = len(image_query)

            if self._contrast_mode == 'supervised':
                label = data['label'].to(self._device)  # (B,)
            elif self._contrast_mode == 'unsupervised':
                label = torch.arange(B).to(self._device)
            else:
                raise NotImplementedError()

            images = torch.cat(
                [image_query, image_key], dim=0).to(self._device)
            # (2*B, C, H, W)
            feats = self._net(images)
            # (B, D)
            q, k = feats.chunk(2, dim=0)

            feats = torch.cat([q, k], dim=0)  # (2 * B, D)
            logits = torch.matmul(feats, feats.T)
            # mask-out self-attention
            logits_mask = 1 - torch.eye(
                B * 2, device=logits.device, dtype=logits.dtype)
            logits = logits * logits_mask

            # (2 * B)
            label = label.repeat(2)

            pred_positive_idx = torch.argmax(logits, dim=1)
            pred_label = label[pred_positive_idx]
            accurate_count = (
                    accurate_count + torch.sum(pred_label == label))
            total_count = total_count + B * 2
        accuracy = accurate_count / total_count
        return {
            'accuracy': accuracy
        }

    def visualize_epoch(
            self, visualization_data_loader: DataLoader
    ) -> Dict[str, Union[torch.Tensor, npt.NDArray[TypeNPDTypeFloat]]]:
        raise NotImplementedError