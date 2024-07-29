#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.
import torch
import argparse
from pathlib import Path
from utils import os_utils, import_utils
import tomllib
from core.datasets.data_module import DataModule
import logging
from tqdm import tqdm
from core.models.protocol import TrainingModelProtocol
from collections import defaultdict
import tensorboardX
from tensorboardX.summary import _clean_tag, convert_to_HWC
from typing import Dict
import numpy as np
import numpy.typing as npt
import imageio.v3 as iio
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

_logger = logging.getLogger(__name__)


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
    output_dir.mkdir(exist_ok=True, parents=True)
    if len(os.listdir(output_dir)) != 0:
        raise RuntimeError('Output directory is not empty.')

    tb_log_dir = output_dir / 'tb_log'
    tb_log_dir.mkdir(exist_ok=True, parents=False)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(name)s] - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'run_log.log'),
            logging.StreamHandler(),
        ]
    )
    _logger.info(
        f'Config loaded from {config_path}.')
    _logger.info(f'Config: \n{config}.')
    _logger.info(f'Created tensorboard log at {tb_log_dir}.')

    n_epochs = config['n_epochs']
    visual_ever_n_epochs = config.get('visual_every_n_epochs', -1)

    model: TrainingModelProtocol = import_utils.get_object_from_config(
        config['model_config'], device=device)

    use_scheduler = config.get('use_lr_scheduler', False)
    min_learning_rate = config.get('min_learning_rate', 0)
    if use_scheduler:
        _logger.info(
            f'Using learning rate scheduler Cosine Annelaing with min_learning_rate {min_learning_rate}.')
        schedulers = []
        for optimizer in model.optimizers:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=min_learning_rate)
            schedulers.append(scheduler)

    data_module = DataModule(**config['data_module_config'])
    if data_module.training_data_loader is None:
        raise RuntimeError('Training data loader is not initialized.')

    tb_writer = tensorboardX.SummaryWriter(str(tb_log_dir))
    image_save_dir = output_dir / 'intermediate_images'

    _logger.info(f'Start training.')
    for epoch in range(1, n_epochs + 1):
        _logger.info(f'Running epoch {epoch}.')
        data_module.set_epoch(epoch)
        _logger.info(f'Training epoch {epoch}.')
        iterator = tqdm(
            data_module.training_data_loader,
            desc='Training',
            total=len(data_module.training_data_loader),
            mininterval=30, maxinterval=60
        )
        n_iter = len(data_module.training_data_loader)
        model.trigger_model(True)
        loss_collector = defaultdict(
            lambda: torch.tensor(
                0., dtype=torch.float32,device=device)
        )
        count = 0
        lr = model.optimizers[0].param_groups[0]['lr']
        tb_writer.add_scalar('lr', lr, epoch)
        for i, data in enumerate(iterator):
            step_loss = model.train_batch(data, epoch=epoch)
            for tag, loss in step_loss.items():
                loss_collector[tag] += loss.mean().detach()
            count += 1
        if use_scheduler:
            for scheduler in schedulers:
                scheduler.step()
        msg = 'Loss:'
        for k, v in loss_collector.items():
            v = v / count
            msg += ' %s: %f' % (k, v.item())
            tb_writer.add_scalar(f'train/{k}', v, epoch)
        msg = msg + '.'
        _logger.info(msg)
        del loss_collector

        _logger.info(f'Evaluating epoch {epoch}.')
        model.trigger_model(False)
        if data_module.test_data_loader is not None:
            msg = 'Evaluation:'
            eval_results = model.evaluate_epoch(
                data_module.test_data_loader)
            for k, v in eval_results.items():
                msg += ' %s: %.6f' % (k, v.item())
                tb_writer.add_scalar(f'eval/{k}', v, epoch)
            msg = msg + '.'
            _logger.info(msg)

        do_visual = True
        if data_module.visualization_data_loader is None:
            do_visual = False
        if visual_ever_n_epochs <= 0:
            do_visual = False
        elif epoch % visual_ever_n_epochs != 0:
            do_visual = False
        else:
            pass
        if do_visual:
            image_save_dir.mkdir(parents=True, exist_ok=True)
            _logger.info(f'Visualizing epoch {epoch}.')
            visual_res: Dict[str, npt.NDArray[np.float32]]
            visual_res = model.visualize_epoch(
                data_module.visualization_data_loader)
            for tag, img in visual_res.items():
                img = (img * 255).astype(np.uint8)
                img = convert_to_HWC(img, 'NHWC')
                tb_writer.add_images(
                    tag, img, epoch, dataformats='HWC')
                tag = _clean_tag(tag)
                file_name = f'e{epoch}_{tag}.jpg'
                save_path = image_save_dir / file_name
                iio.imwrite(save_path, img)
                _logger.info(f'Image wrote to {save_path}')

    _logger.info('Training finished.')
    _logger.info('Try saving model.')
    model.save_model(output_dir, 'ckp')


if __name__ == '__main__':
    main()