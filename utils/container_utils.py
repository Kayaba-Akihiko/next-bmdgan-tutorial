#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Callable, Tuple, Iterable, Any, Dict
from itertools import repeat
import logging

_logger = logging.getLogger(__name__)

def _ntuple(n) -> Callable[[int], Tuple[Any, ...]]:
    def parse(x) -> Tuple[Any, ...]:
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def update_dict_(
        source_dict: Dict, updating_dict: Dict) -> Dict:
    for k, v in updating_dict.items():
        if k in source_dict:
            _logger.info(
                f'Overriding key {k} value from {source_dict[k]} to {v}.')
        source_dict[k] = v
    return source_dict