#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import numpy as np
import os
from typing import Union, Dict, Any
from pathlib import Path

type AnyStr = Union[str, bytes]

type TypeConfig = Dict[str, Any]

type TypeNPDTypeSigned = Union[
    Union[Union[np.short, np.intc], np.int_],
    np.longlong
]

type TypeNPDTypeUnsigned = Union[
    Union[Union[np.ushort, np.uintc], np.uint],
    np.ulonglong
]

type TypeNPDInt = Union[TypeNPDTypeSigned, TypeNPDTypeUnsigned]

type TypeNPDTypeFloat = Union[
    Union[Union[np.half, np.single], np.double],
    np.longdouble,
]

type TypePathLike = Union[Union[AnyStr, os.PathLike[AnyStr]], Path]


