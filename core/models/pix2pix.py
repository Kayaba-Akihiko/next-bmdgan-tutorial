#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.
from pathlib import Path

from .base_model import BaseTrainingModel, BaseTestModel
import torch
from torch.utils.data import DataLoader
from typing import Union, Tuple, Dict, Any, List, Optional, Literal
import logging
from utils import torch_utils
from utils.typing import TypePathLike
from tqdm import tqdm
import numpy.typing as npt
from utils.typing import TypeNPDTypeFloat
from ..networks.nets import Generator, NLayerDiscriminator
from ..networks.losses.gan_loss import GANLoss
import numpy as np
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    UniversalImageQualityIndex,
    VisualInformationFidelity,
)
import torch.nn.functional as F
import polars as pl
import imageio.v3 as iio

_logger = logging.getLogger(__name__)


def _get_net(
        backbone: str,
        image_size: Tuple[int, int],
        image_channels: int,
        pretrain_backbone_load_path: Optional[TypePathLike] = None,
        pretrain_backbone_strict_load=True,
):
    valid_backbones = Generator.supported_backbones
    if backbone not in valid_backbones:
        raise NotImplementedError(
            f'Unknown backbone {backbone}. '
            f'The supported backbones are {valid_backbones}.'
        )
    net = Generator(
        in_channels=image_channels,
        image_size=image_size,
        out_channels=image_channels,
        backbone_cfg={'arch': backbone},
        pretrain_backbone_load_path=pretrain_backbone_load_path,
        pretrain_backbone_strict_load=pretrain_backbone_strict_load,
    )
    return net


class Training(BaseTrainingModel):

    def __init__(
            self,
            image_channels: int,
            image_size: Union[Tuple[int, int], int],
            backbone: str,
            learning_rate: float = 2e-4,
            lambda_gan=1.,
            lambda_recon=100.,
            recon_criterion: Literal['l1', 'l2'] = 'l1',
            device=torch.device('cpu'),
            pretrain_backbone_load_path: Optional[TypePathLike] = None,
            pretrain_backbone_strict_load=True,
    ):
        super().__init__(
            image_channels=image_channels,
            image_size=image_size,
            device=device,
        )

        if lambda_recon <= 0:
            raise ValueError(
                f'lambda_recon {lambda_recon} must be positive.')

        self._lambda_gan = lambda_gan
        self._lambda_recon = lambda_recon

        self._backbone_name = backbone
        net = _get_net(
            self._backbone_name,
            self._image_size,
            self._image_channels,
            pretrain_backbone_load_path=pretrain_backbone_load_path,
            pretrain_backbone_strict_load=pretrain_backbone_strict_load,
        ).to(self._device)

        self._net_g = net
        self._register_network('net_g', self._net_g)
        n_train, n_tix, n_total = torch_utils.count_parameters(
            self._net_g)
        _logger.info(
            f"{self._net_g.__class__} has "
            f"{n_total * 1.e-6:.2f} M params "
            f"({n_train * 1.e-6:.2f} trainable)."
        )
        g_optimizer = torch.optim.AdamW(
            self._net_g.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        g_grad_scaler = torch.amp.grad_scaler.GradScaler()

        self._optimizers = [g_optimizer]
        self._grad_scalers = [g_grad_scaler]

        if recon_criterion == 'l1':
            self._crit_recon = torch.nn.L1Loss()
        elif recon_criterion == 'l2':
            self._crit_recon = torch.nn.MSELoss()

        if self._lambda_gan > 0:
            net = NLayerDiscriminator(self._image_channels).to(self._device)
            self._net_d = net
            self._register_network('net_d', self._net_d)
            n_train, n_tix, n_total = torch_utils.count_parameters(
                self._net_d)
            _logger.info(
                f"{self._net_d.__class__} has "
                f"{n_total * 1.e-6:.2f} M params "
                f"({n_train * 1.e-6:.2f} trainable)."
            )
            d_optimizer = torch.optim.AdamW(
                self._net_d.parameters(),
                lr=learning_rate,
                weight_decay=1e-4,
            )
            d_grad_scaler = torch.amp.grad_scaler.GradScaler()
            self._optimizers.append(d_optimizer)
            self._grad_scalers.append(d_grad_scaler)
            self._crit_gan = GANLoss().to(self._device)

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

        source_image = data['source_image'].to(self._device)
        target_image = data['target_image'].to(self._device)
        fake_target_image = self._net_g(source_image)

        loss_log = {}
        loss_g_recon = self._crit_recon(fake_target_image, target_image)
        loss_log['g_recon'] = loss_g_recon.detach()

        loss_g = loss_g_recon * self._lambda_recon
        if self._lambda_gan > 0:
            self._net_d.requires_grad_(False)
            loss_g_gan = self._crit_gan(
                self._net_d(fake_target_image), target_is_real=True)
            loss_log['g_gan'] = loss_g_gan.detach()
            loss_g = loss_g + loss_g_gan * self._lambda_gan
            self._net_d.requires_grad_(True)

            loss_d_real = self._crit_gan(
                self._net_d(target_image), target_is_real=True)
            loss_d_fake = self._crit_gan(
                self._net_d(fake_target_image.detach()), target_is_real=False)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_log['d_gan'] = loss_d.detach()

            return {
                'loss': (loss_g, loss_d),
                'log': loss_log,
            }

        return {
            'loss': (loss_g,),
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

        psnrs = []
        calc_psnr = PeakSignalNoiseRatio(
            data_range=1, reduction='none', dim=(1, 2, 3)
        ).to(self._device)

        for data in iterator:
            source_image = data['source_image'].to(self._device)
            target_image = data['target_image'].to(self._device)
            norm_mean = data['norm_mean'].to(self._device)  # (B, C)
            norm_std = data['norm_std'].to(self._device)  # (B, C)
            fake_target_image = self._net_g(source_image)

            norm_mean = norm_mean[..., None, None]
            norm_std = norm_std[..., None, None]

            target_image = target_image * norm_std + norm_mean
            target_image = target_image.clamp_(0., 1.)
            fake_target_image = fake_target_image * norm_std + norm_mean
            fake_target_image = fake_target_image.clamp_(0., 1.)

            psnr = calc_psnr(fake_target_image, target_image).view(-1)
            psnrs.append(psnr.cpu())

        psnrs = torch.cat(psnrs, dim=0)
        return {
            'psnr': psnrs.mean()
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
            'source': [],
            'target': [],
            'fake_target': [],
        }

        for data in iterator:
            source_image = data['source_image'].to(self._device)
            target_image = data['target_image'].to(self._device)
            norm_mean = data['norm_mean'].to(self._device)  # (B, C)
            norm_std = data['norm_std'].to(self._device)  # (B, C)
            fake_target_image = self._net_g(source_image)

            norm_mean = norm_mean[..., None, None]
            norm_std = norm_std[..., None, None]

            source_image = source_image * norm_std + norm_mean
            source_image = source_image.clamp_(0., 1.)
            target_image = target_image * norm_std + norm_mean
            target_image = target_image.clamp_(0., 1.)
            fake_target_image = fake_target_image * norm_std + norm_mean
            fake_target_image = fake_target_image.clamp_(0., 1.)

            image_collection['source'].append(
                source_image.cpu())
            image_collection['target'].append(
                target_image.cpu())
            image_collection['fake_target'].append(
                fake_target_image.cpu())

        resolution = max(self._image_size)
        pad_length = max(round(resolution * 0.02), 1)
        pads = (
            (0, 0),
            (pad_length, pad_length),
            (pad_length, pad_length),
            (0, 0)
        )
        for tag in image_collection:
            # (N, C, H, W)
            images = torch.cat(image_collection[tag], dim=0).numpy()
            # (N, H, W, C)
            images = np.einsum('nchw->nhwc', images)
            images = np.pad(
                images, pads, mode='constant', constant_values=1.)
            image_collection[tag] = images
        return image_collection


class TestEvalImage(BaseTestModel):
    def __init__(
            self,
            image_channels: int,
            image_size: Union[Tuple[int, int], int],
            backbone: str,
            device=torch.device('cpu'),
            eval_file_name='result'
    ):
        super().__init__(
            image_channels=image_channels,
            image_size=image_size,
            device=device,
        )
        self._backbone_name = backbone
        self._eval_file_name = eval_file_name


        net = _get_net(
            self._backbone_name,
            self._image_size,
            self._image_channels,
        ).to(self._device)
        self._net_g = net
        self._register_network('net_g', self._net_g)
        n_train, n_tix, n_total = torch_utils.count_parameters(
            self._net_g)
        _logger.info(
            f"{self._net_g.__class__} has "
            f"{n_total * 1.e-6:.2f} M params "
            f"({n_train * 1.e-6:.2f} trainable)."
        )

    @torch.no_grad()
    def test_and_save(
            self,
            test_data_loader: DataLoader,
            save_dir: Path,
    ) -> None:
        if test_data_loader.batch_size != 1:
            raise NotImplementedError(
                'Only support batch size of 1.')

        save_path = save_dir / f'{self._eval_file_name}.csv'
        if save_path.exists():
            raise RuntimeError(f'{save_path} already exists.')

        calc_psnr = PeakSignalNoiseRatio(
            data_range=1).to(self._device)
        calc_ssim = StructuralSimilarityIndexMeasure(
            data_range=1).to(self._device)
        calc_uiq = UniversalImageQualityIndex().to(self._device)
        calc_vif = VisualInformationFidelity().to(self._device)

        eval_results = []
        n_iter = len(test_data_loader)
        iterator = tqdm(
            test_data_loader,
            desc='Evaluating',
            total=n_iter,
            mininterval=30, maxinterval=60,
        )
        for data in iterator:
            batch_source = data['source_image'].to(self._device)
            batch_original_target = data['original_target_image']
            batch_original_target = batch_original_target.to(
                self._device)

            # (B, C)
            batch_norm_mean = data['norm_mean'].to(self._device)
            batch_norm_std = data['norm_std'].to(self._device)

            batch_sample_id = data['sample_id']  # (B,
            batch_fake_target = self._net_g(batch_source)

            batch_norm_mean = batch_norm_mean[..., None, None]
            batch_norm_std = batch_norm_std[..., None, None]

            # denormalization
            batch_fake_target =(
                batch_fake_target * batch_norm_std + batch_norm_mean)
            batch_fake_target = batch_fake_target.clamp_(0., 1.)
            # batch_original_target should be already denormalized.

            B = len(batch_source) # should be always 1

            for i in range(B):
                sample_id = batch_sample_id[i]
                # (1, C, H, W)
                original_target = batch_original_target[i: i + 1]
                _, _, H, W = original_target.shape
                fake_target = batch_fake_target[i: i + 1]
                # (1, C, H, W)
                fake_target = F.interpolate(
                    fake_target,
                    (H, W),
                    mode='bilinear',
                    align_corners=True,
                )
                psnr = calc_psnr(fake_target, original_target)
                ssim = calc_ssim(fake_target, original_target)
                uiq = calc_uiq(fake_target, original_target)
                vif = calc_vif(fake_target, original_target)

                # fake_target = fake_target.cpu().numpy()[0]
                # fake_target = np.einsum(
                #     'ijk->jki', fake_target)
                # fake_target = (fake_target * 255).astype(np.uint8)
                # iio.imwrite(save_dir / f'{sample_id}.jpg', fake_target)

                data = {
                    'sample_id': sample_id,
                    'psnr': psnr,
                    'ssim': ssim,
                    'uiq': uiq,
                    'vif': vif,
                }
                eval_results.append(data)

        eval_df = pl.DataFrame(eval_results)
        eval_df = eval_df.sort(by=['sample_id'])
        save_dir.mkdir(exist_ok=True, parents=True)
        eval_df.write_csv(save_path)
        _logger.info(f'Evaluation results saved to {save_path}.')
