#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import torch
import torch.nn as nn
from typing import AnyStr, Any, Dict, Tuple, List, Union, Optional
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from timm.models.layers import trunc_normal_
from collections import OrderedDict
from .typing import TypePathLike
import logging
import re

_logger = logging.getLogger(__name__)


def load_network_by_dict(
        net: nn.Module,
        params_dict: Dict[str, Any],
        strict=True,
        log=True,
) -> Tuple[List[str], List[str]]:
    if strict:
        return net.load_state_dict(params_dict, strict=strict)
    try:
        missing, unexpected = net.load_state_dict(
            params_dict, strict=strict)
    except RuntimeError:
        loaded = []
        model_dict = net.state_dict()
        for key, value in params_dict.items():
            if key in model_dict:
                if model_dict[key].shape == value.shape:
                    model_dict[key] = value
                    loaded.append(key)
        loaded_keys = set(loaded)
        missing = list(set(model_dict.keys()) - loaded_keys)
        unexpected = list(set(params_dict.keys()) - loaded_keys)
        net.load_state_dict(OrderedDict(model_dict))
    n_total = len(net.state_dict())
    n_missing = len(missing)
    n_unexpect = len(unexpected)
    n_loaded = n_total - n_missing
    if log:
        if n_missing > 0:
            _logger.warning(
                f"Missed {n_missing}/{n_total} keys.")
            _logger.warning(f"Missing keys: {missing}")
        if n_unexpect > 0:
            _logger.warning(
                f"Encountered {n_unexpect} unexpected keys.")
            _logger.warning(f"Unexpected keys: {unexpected}")
        _logger.info(
            f"Model {net.__class__.__name__} ({n_loaded}/{n_total}).")
    return missing, unexpected


def load_network_by_path(
        net: torch.nn.Module, path: TypePathLike, strict=True
) -> Tuple[List[str], List[str]]:
    missing, unexpected = load_network_by_dict(
        net, torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ), strict=strict, log=False)
    n_total = len(net.state_dict())
    n_missing = len(missing)
    n_unexpect = len(unexpected)
    n_loaded = n_total - n_missing

    if n_missing > 0:
        _logger.warning(
            f'Missed {n_missing}/{n_total} keys '
            f'when loading {net.__class__.__name__} from {path}.')
    if n_unexpect > 0:
        _logger.warning(
            f'Encountered {n_unexpect} unexpected keys '
            f'when loading {net.__class__.__name__} from {path}.')
    _logger.info(
        f'Model {net.__class__.__name__} ({n_loaded}/{n_total}) '
        f'loaded from {path}.')
    return missing, unexpected


def set_requires_grad(
        nets: Union[torch.nn.Module, List[torch.nn.Module]],
        requires_grad = False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def count_parameters(module: nn.Module) -> Tuple[int, int, int]:
    state = module.training
    n_trainable = 0
    n_fixed = 0
    module.train(True)
    for p in module.parameters():
        if p.requires_grad:
            n_trainable += p.numel()
        else:
            n_fixed += p.numel()
    module.train(state)
    return n_trainable, n_fixed, n_trainable + n_fixed