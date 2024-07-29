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
from ..networks.nets import Generator
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio

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

        valid_backbones = Generator.supported_backbones
        if backbone not in valid_backbones:
            raise NotImplementedError(
                f'Unknown backbone {backbone}. '
                f'The supported backbones are {valid_backbones}.'
            )
        self._backbone_name = backbone

        net = Generator(
            in_channels=self._image_channels,
            image_size=self._image_size,
            out_channels=self._image_channels,
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

        h, w = self._image_size

        if h != w:
            raise NotImplementedError(self._image_size)

        self._n_patches_y = 16
        self._n_patches_x = 16
        self._n_patches = self._n_patches_x * self._n_patches_y
        self._mask_ratio = 0.75
        self._len_keep = round(self._n_patches * (1 - self._mask_ratio))
        self._patch_size_y = h // self._n_patches_y
        if h % self._n_patches_y != 0:
            raise ValueError(
                f'Patch size must be divisible by {self._n_patches_y}')
        self._patch_size_x = w // self._n_patches_x
        if w % self._n_patches_x != 0:
            raise ValueError(
                f'Patch size must be divisible by {self._n_patches_x}')

        self._optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self._grad_scaler = torch.amp.grad_scaler.GradScaler()
        self._optimizers = [self._optimizer]
        self._grad_scalers = [self._grad_scaler]

        self._crit = torch.nn.MSELoss()

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
            x = patchify_image(image, self._n_patches_y, self._n_patches_x)
            B, L, D = x.shape

            mask_res = self._random_masking(x)
            x_masked = mask_res['x_masked']
            ids_noising = mask_res['ids_noising']  # (B, L * 0.75)
            ids_restore = mask_res['ids_restore']
            del mask_res

            noise = torch.randn(
                (B, L - self._len_keep, D),
                dtype=torch.float32,
                device=self._device,
            )

            # (B, L * 0.25, D) c (B, L * 0.75, D) -> (B, L, D)
            x_masked = torch.cat((x_masked, noise), dim=1)
            del noise

            # restore the order
            x_masked = torch.gather(
                x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
            # (B, C, H, W)
            masked_image = unpatchify(
                x_masked,
                self._image_size,
                self._n_patches_y,
                self._n_patches_x,
                self._image_channels
            )
            del x_masked

        # (B, C, H, W)
        pred_image = self._net(masked_image)
        del masked_image

        # (B, L, D)
        pred_x = patchify_image(
            pred_image, self._n_patches_y, self._n_patches_x)

        # (B, L * 0.75, D)
        pred_x = torch.gather(
            pred_x, dim=1, index=ids_noising.unsqueeze(-1).repeat(1, 1, D))
        x = torch.gather(
            x, dim=1, index=ids_noising.unsqueeze(-1).repeat(1, 1, D))

        loss = self._crit(pred_x, x)
        loss_log = {
            'recon_loss': loss.detach()
        }
        return {
            'loss': (loss, ),
            'log': loss_log
        }

    def _random_masking(self, x: torch.Tensor):
        # x: (B, L, D)
        B, L, D = x.shape
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :self._len_keep]
        ids_noising = ids_shuffle[:, self._len_keep:]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :self._len_keep] = 0
        # unshuffle to get the binary mask
        # (B, L)
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # return x_masked, mask, ids_shuffle, ids_restore, ids_keep, ids_masking
        return {
            'x_masked': x_masked,
            'mask': mask,
            'id_shuffle': ids_shuffle,
            'ids_restore': ids_restore,
            'ids_keep': ids_keep,
            'ids_noising': ids_noising
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
            image = data['image'].to(self._device)
            norm_mean = data['norm_mean'].to(self._device)  # (B, C)
            norm_std = data['norm_std'].to(self._device)  # (B, C)
            x = patchify_image(
                image, self._n_patches_y, self._n_patches_x)
            B, L, D = x.shape

            mask_res = self._random_masking(x)
            x_masked = mask_res['x_masked']
            ids_noising = mask_res['ids_noising']  # (B, L * 0.75)
            ids_restore = mask_res['ids_restore']
            del mask_res

            noise = torch.randn(
                (B, L - self._len_keep, D),
                dtype=torch.float32,
                device=self._device,
            )

            # (B, L * 0.25, D) c (B, L * 0.75, D) -> (B, L, D)
            x_masked = torch.cat((x_masked, noise), dim=1)
            del noise

            # restore the order
            x_masked = torch.gather(
                x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
            # (B, C, H, W)
            masked_image = unpatchify(
                x_masked,
                self._image_size,
                self._n_patches_y,
                self._n_patches_x,
                self._image_channels
            )
            del x_masked

            # (B, C, H, W)
            pred_image = self._net(masked_image)
            del masked_image

            # (B, L, D)
            pred_x = patchify_image(
                pred_image, self._n_patches_y, self._n_patches_x)

            # (B, L * 0.75, D)
            pred_x = torch.gather(
                pred_x, dim=1, index=ids_noising.unsqueeze(-1).repeat(1, 1, D))
            pred_x = rearrange(
                pred_x, 'b l (psh psw c) -> b c l (psh psw)',
                c=self._image_channels,
                psh=self._patch_size_y,
                psw=self._patch_size_x,
            )

            x = torch.gather(
                x, dim=1, index=ids_noising.unsqueeze(-1).repeat(1, 1, D))
            x = rearrange(
                x, 'b l (psh psw c) -> b c l (psh psw)',
                c=self._image_channels,
                psh=self._patch_size_y,
                psw=self._patch_size_x,
            )

            norm_mean = norm_mean[..., None, None]  # (B, C, 1, 1)
            norm_std = norm_std[..., None, None]

            pred_x = pred_x * norm_std + norm_mean
            pred_x = pred_x.clamp_(0., 1.)
            x = x * norm_std + norm_mean
            x = x.clamp_(0., 1.)
            psnr = calc_psnr(pred_x, x).view(-1)  # (B,)
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
            'image': [],
            'pred_image': [],
            'masked_image': [],
        }
        for data in iterator:
            # (B, L, D)
            image = data['image'].to(self._device)
            norm_mean = data['norm_mean'].to(self._device)  # (B, C)
            norm_std = data['norm_std'].to(self._device)  # (B, C)
            x = patchify_image(
                image, self._n_patches_y, self._n_patches_x)
            B, L, D = x.shape

            mask_res = self._random_masking(x)
            x_masked = mask_res['x_masked']
            ids_noising = mask_res['ids_noising']  # (B, L * 0.75)
            ids_restore = mask_res['ids_restore']
            mask = mask_res['mask']  # (B, L)
            del mask_res

            noise = torch.randn(
                (B, L - self._len_keep, D),
                dtype=torch.float32,
                device=self._device,
            )

            # (B, L * 0.25, D) c (B, L * 0.75, D) -> (B, L, D)
            x_masked = torch.cat((x_masked, noise), dim=1)
            del noise

            # restore the order
            x_masked = torch.gather(
                x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
            # (B, C, H, W)
            masked_image = unpatchify(
                x_masked,
                self._image_size,
                self._n_patches_y,
                self._n_patches_x,
                self._image_channels
            )
            del x_masked

            # (B, C, H, W)
            pred_image = self._net(masked_image)

            norm_mean = norm_mean[..., None, None]  # (B, C, 1, 1)
            norm_std = norm_std[..., None, None]

            image = image * norm_std + norm_mean
            image = image.clamp_(0., 1.)

            pred_image = pred_image * norm_std + norm_mean
            pred_image = pred_image.clamp_(0., 1.)

            # (B, C, H, W)
            masked_image = masked_image * norm_std + norm_mean
            masked_image = masked_image.clamp_(0., 1.)

            # (B, L, D)
            mask = mask[..., None].repeat(1, 1, D)
            # (B, C, H, W)
            mask_image = unpatchify(  # 1 means noise
                mask,
                self._image_size,
                self._n_patches_y,
                self._n_patches_x,
                self._image_channels
            ).to(torch.float32)
            # replace noise with constant value
            masked_image = (
                    masked_image * (1. - mask_image) + 0.5 * mask_image)

            image_collection['image'].append(image.cpu())
            image_collection['pred_image'].append(pred_image.cpu())
            image_collection['masked_image'].append(masked_image.cpu())

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


def patchify_image(
        image: torch.Tensor, n_patches_y: int, n_patches_x: int
) -> torch.Tensor:
    # (B, C, H, W)
    B, C, H, W = image.shape
    # Omitting checking
    # if H % n_patches_y != 0:
    #     raise NotImplementedError(f'{image.shape} {n_patches_y}')
    # if W % n_patches_x != 0:
    #     raise NotImplementedError(f'{image.shape} {n_patches_x}')
    patch_size_y = H // n_patches_y
    patch_size_x = W // n_patches_x

    x = rearrange(
        image,
        'b c (nph psh) (npw psw) -> b (nph npw) (psh psw c)',
        nph=n_patches_y, psh=patch_size_y,
        npw=n_patches_x, psw=patch_size_x
    )
    return x


def unpatchify(
        x: torch.Tensor,
        image_size: Tuple[int, int],
        n_patches_y: int,
        n_patches_x: int,
        image_channels=3,

) -> torch.Tensor:
    #  x (B, L, D)

    H, W = image_size
    # Omitting check
    # if n_patches_y * n_patches_x != L:
    #     raise NotImplementedError(f'{x.shape}')
    # if H % n_patches_y != 0:
    #     raise NotImplementedError(f'{image_size} {n_patches_y}')
    # if W % n_patches_x != 0:
    #     raise NotImplementedError(f'{image_size} {n_patches_x}')
    patch_size_y = H // n_patches_y
    patch_size_x = W // n_patches_x

    image = rearrange(
        x, 'b (nph npw) (psh psw c) -> b c (nph psh) (npw psw)',
        c=image_channels,
        nph=n_patches_y, psh=patch_size_y,
        npw=n_patches_x, psw=patch_size_x
    )
    return image