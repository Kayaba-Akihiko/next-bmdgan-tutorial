#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import torch
import argparse
from utils import os_utils, import_utils
import tomllib
from core.datasets.data_module import DataModule
import logging
from core.models.protocol import TrainingModelProtocol, TestModelProtocol

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

    model: TestModelProtocol = import_utils.get_object_from_config(
        config['model_config'], device=device)

    pretrain_load_dir = os_utils.format_path_str(
        config['pretrain_load_dir'])
    pretrain_load_prefix = 'ckp'

    data_module = DataModule(**config['data_module_config'])
    if data_module.test_data_loader is None:
        raise RuntimeError('Test data loader is not initialized.')

    _logger.info('Loading model.')
    model.load_model(
        load_dir=pretrain_load_dir,
        prefix=pretrain_load_prefix,
        strict=True
    )
    _logger.info(
        f'Model loaded from {pretrain_load_dir} '
        f'with prefix {pretrain_load_prefix}.'
    )

    _logger.info(f'Start evaluation with save dir {output_dir}.')
    model.test_and_save(data_module.test_data_loader, save_dir=output_dir)


if __name__ == '__main__':
    main()