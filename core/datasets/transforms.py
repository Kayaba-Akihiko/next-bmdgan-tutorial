#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import List, Any
import torchvision.transforms.v2 as v2


class RandomVerticalFlip(v2.RandomVerticalFlip):
    def _needs_transform_list(
            self, flat_inputs: List[Any]) -> List[bool]:
        return [True] * len(flat_inputs)

class RandomHorizontalFlip(v2.RandomHorizontalFlip):
    def _needs_transform_list(
            self, flat_inputs: List[Any]) -> List[bool]:
        return [True] * len(flat_inputs)