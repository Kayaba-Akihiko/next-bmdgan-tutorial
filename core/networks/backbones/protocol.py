#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import Protocol, Tuple, Union, Dict, List
from abc import abstractmethod

import torch.nn
import torch.nn as nn

TypeBuildResults = Dict[str, Union[nn.Module, Tuple[int, int, int, int]]]


class BackboneBuilderProtocol(Protocol):

    def __init__(
            self,
            in_channels: int,
            image_size: Tuple[int, int],
            **kwargs,
    ):
        self._in_channels = in_channels
        self._image_size = image_size
        self._kwargs = kwargs

    @abstractmethod
    def build(self) -> TypeBuildResults:
        raise NotImplementedError