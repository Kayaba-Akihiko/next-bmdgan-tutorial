#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Union, Optional, Tuple, Sequence
import torch
import math


def get_intersection_and_sum(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
)-> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.shape != y.shape:
        raise RuntimeError(
            f'{x.shape} and {y.shape} do not have the same size.')
    if torch.min(x) < 0 or torch.max(x) > 1:
        raise RuntimeError(
            f'Only support binary values '
            f'but got {torch.min(x)} {torch.max(x)}.')
    if torch.min(y) < 0 or torch.max(y) > 1:
        raise RuntimeError(
            f'Only support binary values '
            f'but got {torch.min(y)} {torch.max(y)}.')

    intersection = torch.sum(torch.mul(x, y), dim=dim, keepdim=keepdim)
    sum_ = (
            torch.sum(x, dim=dim, keepdim=keepdim)
            + torch.sum(y, dim=dim, keepdim=keepdim)
    )
    mask = (sum_ > 0.)
    return intersection, sum_, mask


def dice_score(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
) -> torch.Tensor:
    intersection, sum_, mask = get_intersection_and_sum(
        x=x, y=y, dim=dim, keepdim=keepdim)
    dice = torch.ones_like(intersection)
    dice[mask] = torch.div(intersection[mask] * 2., sum_[mask])
    return dice


def jaccard_index(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
) -> torch.Tensor:
    intersection, sum_, mask = get_intersection_and_sum(
        x=x, y=y, dim=dim, keepdim=keepdim)
    jac = torch.ones_like(intersection)
    valid_intersection = intersection[mask]
    jac[mask] = torch.div(
        valid_intersection, sum_[mask] - valid_intersection)
    return jac


def dice_jaccard(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    intersection, sum_, mask = get_intersection_and_sum(
        x=x, y=y, dim=dim, keepdim=keepdim)

    dice = torch.ones_like(intersection)
    dice[mask] = torch.div(intersection[mask] * 2., sum_[mask])

    jac = torch.ones_like(intersection)
    valid_intersection = intersection[mask]
    jac[mask] = torch.div(
        valid_intersection, sum_[mask] - valid_intersection)
    return dice, jac