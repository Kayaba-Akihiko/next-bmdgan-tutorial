#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from utils import os_utils, torch_utils
import tomllib
import logging
from typing import Tuple, Optional, List, Dict, Any
from core.networks.nets import Generator
from core.datasets.oxford_iiit_pet.pet_decomposition import TestDataset
import platform
from tqdm import tqdm
import numpy as np
from pathlib import Path
import numpy.typing as npt
import imageio.v3 as iio


_logger = logging.getLogger(__name__)


def _get_net(
        backbone: str,
        image_size: Tuple[int, int],
        image_channels: int,
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
    )
    return net

def main():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config_path', type=str, required=True)
    parser.add_argument(
        '--device', type=str, default='cuda')

    opt = parser.parse_args()
    config_path = os_utils.format_path_str(opt.config_path)
    device = torch.device(opt.device)

    with config_path.open(mode='rb') as f:
        config = tomllib.load(f)

    output_dir = os_utils.format_path_str(config['output_dir'])
    data_root = os_utils.format_path_str(config['data_root'])
    allow_samples: List[str] = config['allow_samples']
    net_configs: List[Dict[str, Any]] = config['net_configs']

    n_workers = config.get(
        'n_workers', os_utils.get_max_n_worker())
    if n_workers > os_utils.get_max_n_worker():
        n_workers = os_utils.get_max_n_worker()

    output_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(name)s] - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'run_log.log'),
            logging.StreamHandler(),
        ]
    )

    """
    net_config: {
        name: 256_convnext_tiny,
        backbone: convnextv2_tiny,
        image_size: (256, 256),
        weights_path: workspace_dir/train/256_convnext_tiny/ckp_net_g.pt
    }
    """

    # Check resolutions
    resolutions = set()
    for net_config in net_configs:
        image_size = net_config['image_size']
        H, W = image_size
        if H != W:
            raise NotImplementedError('H must be equal to W.')
        resolutions.add(H)

    resolution_dataloaders = {}
    pin_memory = True
    if platform.system() == "Windows":
        pin_memory = False
    for resolution in resolutions:
        dataset = TestDataset(
            data_root=data_root,
            image_size=(resolution, resolution),
            mode='pet',
            preload=True,
            n_workers=n_workers,
            allow_samples=allow_samples,
            return_original_image=True,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=pin_memory,
        )
        resolution_dataloaders[resolution] = data_loader

    for net_config in net_configs:
        name = net_config['name']
        image_size = net_config['image_size']
        resolution = image_size[0]
        backbone = net_config['backbone']
        weights_path = os_utils.format_path_str(
            net_config['weights_path'])
        if not weights_path.exists():
            _logger.info(
                'Weights for {name} not found at {weights_path}. Skip.')
        net = _get_net(
            backbone=backbone, image_size=image_size, image_channels=3,
        ).to(device)
        net.eval()
        net.requires_grad_(False)
        _logger.info(f'Loading weights for {name} from {weights_path}.')
        torch_utils.load_network_by_path(net, weights_path, True)

        data_loader = resolution_dataloaders[resolution]
        iterator = tqdm(data_loader, desc=f'Visualizing for {name}.')

        for data in iterator:
            batch_source = data['source_image'].to(device)
            batch_original_target = data['original_target_image']
            batch_original_source = data['original_source_image']
            batch_norm_mean = data['norm_mean'].to(device)
            batch_norm_std = data['norm_std'].to(device)
            batch_sample_id = data['sample_id']  # (B,
            batch_fake_target = net(batch_source)

            batch_norm_mean = batch_norm_mean[..., None, None]
            batch_norm_std = batch_norm_std[..., None, None]
            # denormalization
            batch_fake_target = (
                    batch_fake_target * batch_norm_std + batch_norm_mean)
            batch_fake_target = batch_fake_target.clamp_(0., 1.)
            # batch_original_target should be already denormalized.

            B = len(batch_source)  # should be always 1
            for i in range(B):
                sample_id = batch_sample_id[i]
                # (1, C, H, W)
                original_target = batch_original_target[i: i + 1]
                original_source = batch_original_source[i: i + 1]
                _, _, H, W = original_target.shape
                fake_target = batch_fake_target[i: i + 1]
                # (1, C, H, W)
                fake_target = F.interpolate(
                    fake_target,
                    (H, W),
                    mode='bilinear',
                    align_corners=True,
                )
                sample_save_dir = output_dir / sample_id
                sample_save_dir.mkdir(exist_ok=True, parents=True)
                source_save_path = sample_save_dir / 'source.png'
                target_save_path = sample_save_dir / 'target.png'
                fake_save_path = sample_save_dir / f'target_by_{name}.png'

                # (C, H, W)
                fake_target = fake_target[0].cpu().numpy()
                fake_target = np.einsum(
                    'ijk->jki', fake_target)
                _save_image(fake_save_path, fake_target)
                # _logger.info(
                #     f'Fake target image saved to {fake_save_path}.')
                if not source_save_path.exists():
                    # (C, H, W)
                    original_source = original_source[0].cpu().numpy()
                    original_source = np.einsum(
                        'ijk->jki', original_source)
                    _save_image(source_save_path, original_source)
                    # _logger.info(
                    #     f'Source image saved to {source_save_path}.')
                if not target_save_path.exists():
                    # (C, H, W)
                    original_target = original_target[0].cpu().numpy()
                    original_target = np.einsum(
                        'ijk->jki', original_target)
                    _save_image(target_save_path, original_target)
                    # _logger.info(
                    #     f'Target image saved to {target_save_path}.')


def _save_image(save_path: Path, image: npt.NDArray[np.float32]):
    image = np.round(image * 255.).astype(np.uint8)
    iio.imwrite(save_path, image)


if __name__ == '__main__':
    main()