#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from tqdm import tqdm

import multiprocessing.pool as mpp
from typing import Any, Callable, Generator, Optional, Dict, Union
from collections.abc import Iterable, Sequence


class MultiProcessingAgent:
    def __init__(self, pool_class: str = 'torch', contex=None):
        if pool_class == 'torch':
            import torch.multiprocessing as mp
            from torch.multiprocessing import Pool
        else:
            import multiprocessing as mp
            from multiprocessing import Pool

        if contex is not None:
            Pool = mp.get_context(contex).Pool

        self._pool = Pool

    @staticmethod
    def _function_proxy(fun: Callable, kwargs: Dict[str, Any]) -> Any:
        return fun(**kwargs)

    def run(self,
            args: Union[Sequence[Any], Iterable[Any]],
            n_workers: int,
            func: Optional[Callable] = None,
            desc: Optional[str] = None,
            mininterval: Optional[float] = None,
            maxinterval: Optional[float] = None,
            total: Optional[int] = None,
            progress_bar=True,):
        tqdm_args = {"desc": desc}
        if total is not None:
            tqdm_args['total'] = total
        if isinstance(args, Sequence):
            tqdm_args['total'] = len(args)
        if mininterval is not None:
            tqdm_args["mininterval"] = mininterval
        if maxinterval is not None:
            tqdm_args["maxinterval"] = maxinterval
        exec_fun = self._function_proxy if func is None else func
        if n_workers > 0:
            with self._pool(n_workers) as pool:
                provider = pool.istarmap(exec_fun, iterable=args)
                if progress_bar:
                    provider = tqdm(provider, **tqdm_args)
                for data in provider:
                    yield data
        else:
            iterator = args
            if progress_bar:
                iterator = tqdm(args, **tqdm_args)
            for item in iterator:
                yield exec_fun(*item)


def istarmap(
        self: mpp.Pool, func: Callable, iterable: Iterable, chunksize=1
) -> Generator[Any, Any, None]:
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    task_data = (
        self._guarded_task_generation(
            result._job, mpp.starmapstar, task_batches),
        result._set_length)
    self._taskqueue.put(task_data)
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap