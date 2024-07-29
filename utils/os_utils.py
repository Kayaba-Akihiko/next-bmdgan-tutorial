#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import os
from os import scandir
import numpy as np
import random
import torch
from .typing import AnyStr, TypePathLike
from typing import Callable, Iterable, Union, Optional, Tuple, List
import re
from pathlib import Path


_max_num_worker_suggest = 0
if hasattr(os, 'sched_getaffinity'):
    try:
        _max_num_worker_suggest = len(os.sched_getaffinity(0))
    except Exception:
        pass
if _max_num_worker_suggest == 0:
    # os.cpu_count() could return Optional[int]
    # get cpu count first and check None in order to satify mypy check
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        _max_num_worker_suggest = cpu_count

if "SLURM_CPUS_PER_TASK" in os.environ:
    _max_num_worker_suggest = int(os.environ["SLURM_CPUS_PER_TASK"])


def get_max_n_worker() -> int:
    return _max_num_worker_suggest


def set_seed(seed: int, cuda_deterministic=False) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def format_path_str(path: AnyStr, src_sep='\\', tar_ep='/') -> Path:
    if isinstance(path, Path):
        return path
    res = str(path).replace(src_sep, tar_ep)
    return Path(res)


def scan_dirs_for_file(
        paths: Union[
            Union[TypePathLike, List[TypePathLike]],
            Tuple[TypePathLike, ...]
        ],
        name_re_pattern: Optional[Union[str, re.Pattern]] = None
) -> Iterable[os.DirEntry]:
    def allow_file(entry: os.DirEntry) -> bool:
        return entry.is_file()
    return _scan_dirs(
        paths=paths, allow_fun=allow_file, name_re_pattern=name_re_pattern)


def scan_dirs_for_folder(
        paths: Union[
            Union[TypePathLike, List[TypePathLike]],
            Tuple[TypePathLike, ...]
        ],
        name_re_pattern: Optional[Union[str, re.Pattern]] = None
) -> Iterable[os.DirEntry]:
    def allow_dir(entry: os.DirEntry) -> bool:
        return entry.is_dir()
    return _scan_dirs(
        paths=paths, allow_fun=allow_dir, name_re_pattern=name_re_pattern)


def _scan_dirs(
        paths: TypePathLike | list[TypePathLike] | tuple[TypePathLike, ...],
        allow_fun: Callable[[os.DirEntry], bool],
        name_re_pattern: Optional[Union[str, re.Pattern]] = None
) -> Iterable[os.DirEntry]:
    if not isinstance(paths, list) and not isinstance(paths, tuple):
        paths = [paths]
    if name_re_pattern is None:
        name_re_pattern = re.compile(".*")
    elif isinstance(name_re_pattern, str):
        name_re_pattern = re.compile(name_re_pattern)

    if not isinstance(name_re_pattern, re.Pattern):
        raise RuntimeError(
            f"Unsupported type of name_re_pattern ({type(name_re_pattern)}) ")
    for path in paths:
        with scandir(path) as it:
            entry: os.DirEntry
            for entry in it:
                name_match = name_re_pattern.match(entry.name) is not None
                if allow_fun(entry) and name_match:
                    yield entry