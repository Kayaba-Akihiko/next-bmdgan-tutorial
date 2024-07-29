#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import argparse
from utils import os_utils
import polars as pl
from pathlib import Path
from tqdm import tqdm
import logging
import tomllib

_logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config_path', type=str, required=True)
    opt = parser.parse_args()

    config_path = os_utils.format_path_str(opt.config_path)

    with config_path.open(mode='rb') as f:
        config = tomllib.load(f)

    output_dir = os_utils.format_path_str(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    save_path = output_dir / 'summary.csv'

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(name)s] - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'run_log.log'),
            logging.StreamHandler(),
        ]
    )

    method_evaluation_root = os_utils.format_path_str(
        config['method_evaluation_root'])
    pretraining_explain = config.get('pretraining_explain', {})
    if len(pretraining_explain) == 0:
        _logger.info('No pretraining explain config is provided.')

    summary_data = []
    iterator = list(os_utils.scan_dirs_for_folder(
        method_evaluation_root))
    for method_entry in tqdm(iterator, desc='Collecting results'):
        resolution, backbone, *pretraining = method_entry.name.split('_')
        if len(pretraining) == 0:
            pretraining = 'p0'
        else:
            pretraining = '_'.join(pretraining)
            pretraining = pretraining_explain.get(pretraining, pretraining) # remap

        method_eval_load_path = Path(method_entry.path) / 'result.csv'
        if not method_eval_load_path.exists():
            _logger.info(
                f'Found no result file at {method_eval_load_path}. '
                f'Skip.')
        method_eval_df = pl.read_csv(method_eval_load_path)
        method_eval_df = method_eval_df.fill_nan(0)

        data = {
            'resolution': resolution,
            'backbone': backbone,
            'pretraining': pretraining,
        }
        for target in ['psnr', 'ssim', 'uiq', 'vif']:
            vals = method_eval_df.select(target).to_series().to_numpy()
            data[f'{target}_mean'] = vals.mean()
            data[f'{target}_std'] = vals.std(ddof=1)
        summary_data.append(data)
    if len(summary_data) < 0:
        _logger.info('No summary data found.')
        return
    summary_df = pl.DataFrame(summary_data)
    del summary_data
    summary_df = summary_df.sort(
        by=['resolution', 'backbone', 'pretraining']
    )
    summary_df.write_csv(save_path)
    _logger.info(f'Summary saved to {save_path}.')



    pass

if __name__ == '__main__':
    main()