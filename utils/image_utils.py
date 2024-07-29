#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import numpy as np
from typing import Sequence, Union, Optional, Tuple
from .typing import TypeNPDTypeUnsigned, TypeNPDInt
import numpy.typing as npt
import skimage.transform as skt


def resize[T](
        image: npt.NDArray[T],
        output_shape: Tuple[int, int],
        order: int,
        mode='reflect',
        cval=0,
        clip=True,
        preserve_range=True,
        anti_aliasing=True,
        anti_aliasing_sigma: Optional[
            Union[float, Tuple[float, float]]
        ] = None
) -> npt.NDArray[T]:
    if order == 0:
        anti_aliasing = False
    return skt.resize(
        image,
        output_shape=output_shape,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma,
    )


def center_padding[T](
        image: npt.NDArray[T],
        w_to_h_ratio: float,
        mode='constant',
        constant_values: Union[float, Sequence[float]] = 0.,
) -> npt.NDArray[T]:
    H, W = image.shape[: 2]
    pad_width = calc_center_padding(
        (H, W), w_to_h_ratio=w_to_h_ratio)
    return padding(
        image=image,
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_values,
    )

def calc_center_padding(
        image_size: Tuple[int, int], w_to_h_ratio: float
) -> npt.NDArray[TypeNPDTypeUnsigned]:
    H, W = image_size
    image_size_ratio = W / H
    pads = [(0, 0), (0, 0)]
    if image_size_ratio < w_to_h_ratio:
        new_W = H * w_to_h_ratio
        pad_length = new_W - W
        pad_left = round(pad_length / 2)
        pad_right = round(pad_length - pad_left)
        pads[1] = (pad_left, pad_right)
    elif image_size_ratio > w_to_h_ratio:
        new_H = W / w_to_h_ratio
        pad_length = new_H - H
        pad_left = round(pad_length / 2)
        pad_right = round(pad_length - pad_left)
        pads[0] = (pad_left, pad_right)
    return np.asarray(pads)


def padding[T](
        image: npt.NDArray[T],
        pad_width: Union[int, npt.NDArray[TypeNPDInt]],
        mode='constant',
        constant_values: Union[float, Sequence[float]] = 0.
) -> npt.NDArray[T]:
    pad_width = formate_pad_width(
        pad_width=pad_width, image_ndim=image.ndim)
    return np.pad(image, pad_width, mode, constant_values=constant_values)


def formate_pad_width(
        pad_width: Union[int, npt.NDArray[TypeNPDTypeUnsigned]],
        image_ndim: int,
) -> npt.NDArray[TypeNPDTypeUnsigned]:
    if not isinstance(pad_width, np.ndarray):
        assert isinstance(pad_width, int)
        pad_width = np.zeros((2, 2), dtype=int) + pad_width
    else:
        assert pad_width.shape == (2, 2)

    len_pad_width = len(pad_width)
    dim_diff = len_pad_width - image_ndim
    if dim_diff < 0:
        pad_width_add = np.zeros(
            (-dim_diff, 2), dtype=pad_width.dtype)
        pad_width = np.concatenate([pad_width, pad_width_add])
    else:
        assert dim_diff == 0, f'{len_pad_width} {image_ndim}'
    return pad_width