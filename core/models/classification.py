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
from ..networks.nets import Classifier
import logging
from utils import torch_utils
from torchmetrics.classification import MulticlassStatScores
from torchmetrics.classification import MulticlassAUROC
from tqdm import tqdm
import numpy.typing as npt
from utils.typing import TypeNPDTypeFloat, TypePathLike

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

        self._crit_ce = torch.nn.CrossEntropyLoss()

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
        label = data['label'].to(self._device)
        pred_logits = self._net(image)
        loss: torch.Tensor = self._crit_ce(pred_logits, label)
        loss_log: Dict[str, torch.Tensor] = {
            'ce': loss.detach(),
        }
        return {
            'loss': (loss, ),
            'log': loss_log,
        }

    @torch.no_grad()
    def evaluate_epoch(
            self,  test_data_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        confusion_metric = MulticlassStatScores(
            num_classes=self._n_classes, average='none')
        auc_metric = MulticlassAUROC(
            num_classes=self._n_classes, average='none')

        n_iter = len(test_data_loader)
        iterator = tqdm(
            test_data_loader,
            desc='Evaluating',
            total=n_iter,
            mininterval=30, maxinterval=60,
        )
        eps = 1e-8
        pred_labels = []
        pred_probs = []
        labels = []
        for data in iterator:
            image = data['image'].to(self._device)
            label = data['label'].to(self._device)
            pred_logits = self._net(image)
            pred_prob = torch.softmax(pred_logits, dim=1)  # (B, C)
            pred_label = torch.argmax(pred_logits, dim=1, keepdim=False)
            pred_labels.append(pred_label.cpu())
            pred_probs.append(pred_prob.cpu())
            labels.append(label.cpu())

        pred_labels = torch.cat(pred_labels, dim=0)
        pred_probs = torch.cat(pred_probs, dim=0)
        labels = torch.cat(labels, dim=0)

        auc = auc_metric(pred_probs, labels)
        states = confusion_metric(pred_labels, labels)  # (n_classes, 5)
        m_states = torch.sum(states, dim=0)  # (5, )

        tp, fp, tn, fn, sup = states.chunk(5, dim=1)
        # sup = tp + fn
        tp, fp, tn, fn, sup = (
            tp.view(self._n_classes),
            fp.view(self._n_classes),
            tn.view(self._n_classes),
            fn.view(self._n_classes),
            sup.view(self._n_classes),
        )

        f1 = (2 * tp) / (2 * tp + fp + fn)
        acc = (tp + tn) / (tp + fp + tn + fn)
        preci = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        speci = tn / (tn + fp + eps)

        m_tp, m_fp, m_tn, m_fn, m_sup = m_states
        results = dict(
            macro_auc=auc.mean(),
            macro_f1=f1.mean(),
            macro_accuracy=acc.mean(),
            macro_precision=preci.mean(),
            macro_recall=recall.mean(),
            macro_specificity=speci.mean(),
            micro_f1=((2 * m_tp) / (2 * m_tp + m_fp + m_fn)),
            micro_accuracy=(m_tp + m_tn) / (m_tp + m_fp + m_tn + m_fn),
            micro_precision=m_tp / (m_tp + m_fp + eps),
            micro_recall=m_tp / (m_tp + m_fn + eps),
            micro_specificity=m_tn / (m_tn + m_fp + eps),
        )
        for i, c_name in enumerate(self._class_names):
            results[f'{c_name}_auc'] = auc[i]
            results[f'{c_name}_f1'] = f1[i]
            results[f'{c_name}_accuracy'] = acc[i]
            results[f'{c_name}_precision'] = preci[i]
            results[f'{c_name}_recall'] = recall[i]
            results[f'{c_name}_specificity'] = speci[i]

        return results

    def visualize_epoch(
            self, visualization_data_loader: DataLoader
    ) -> Dict[str, Union[torch.Tensor, npt.NDArray[TypeNPDTypeFloat]]]:
        raise NotImplementedError