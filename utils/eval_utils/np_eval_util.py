#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from ..typing import TypeNPDTypeFloat
from typing import Union, Optional, Tuple, Sequence
import numpy as np
import numpy.typing as npt
import math


def get_intersection_and_sum(
        x: npt.NDArray[TypeNPDTypeFloat],
        y: npt.NDArray[TypeNPDTypeFloat],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
) -> Tuple[
    npt.NDArray[TypeNPDTypeFloat],
    npt.NDArray[TypeNPDTypeFloat],
    npt.NDArray[TypeNPDTypeFloat],
]:
    if x.shape != y.shape:
        raise RuntimeError(
            f'{x.shape} and {y.shape} do not have the same size.')
    if np.min(x) < 0 or np.max(x) > 1:
        raise RuntimeError(
            f'Only support binary values but got {np.min(x)} {np.max(x)}.')
    if np.min(y) < 0 or np.max(y) > 1:
        raise RuntimeError(
            f'Only support binary values but got {np.min(y)} {np.max(y)}.')

    intersection = np.sum(np.multiply(x, y), axis=axis, keepdims=keepdims)
    sum_ = (
            np.sum(x, axis=axis, keepdims=keepdims)
            + np.sum(y, axis=axis, keepdims=keepdims)
    )
    mask = (sum_ > 0.)
    return intersection, sum_, mask


def dice_score(
        x: npt.NDArray[TypeNPDTypeFloat],
        y: npt.NDArray[TypeNPDTypeFloat],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
) -> Union[float, npt.NDArray[TypeNPDTypeFloat]]:
    intersection, sum_, mask = get_intersection_and_sum(
        x=x, y=y, axis=axis, keepdims=keepdims)
    dice = np.ones_like(intersection)
    dice[mask] = np.divide(intersection[mask] * 2., sum_[mask])
    return dice


def jaccard_index(
        x: npt.NDArray[TypeNPDTypeFloat],
        y: npt.NDArray[TypeNPDTypeFloat],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
) -> Union[float, npt.NDArray[TypeNPDTypeFloat]]:
    # x: (B, C, H*W)
    # Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    intersection, sum_, mask = get_intersection_and_sum(
        x=x, y=y, axis=axis, keepdims=keepdims)
    jac = np.ones_like(intersection)
    valid_intersection = intersection[mask]
    jac[mask] = np.divide(
        valid_intersection, sum_[mask] - valid_intersection)
    return jac
