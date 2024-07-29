#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import torch
from torch import nn
from typing import Literal


class DiceLoss(nn.Module):

    def __init__(
            self,
            reduction: Literal['mean', 'sum', 'none']='mean',
    ):
        super(DiceLoss, self).__init__()
        self.reduction = reduction

        if self.reduction == 'mean':
            self.agg_fun = torch.mean
        elif self.reduction == 'sum':
            self.agg_fun = torch.sum
        elif self.reduction == 'none':
            self.agg_fun = nn.Identity()
        else:
            raise NotImplementedError(f'Unknown reduction {reduction}.')
    def forward(self, input, target):
        # input, target (B, C, ...)
        input = torch.flatten(input, start_dim=2)  # (B, C, N)
        target = torch.flatten(target, start_dim=2) # (B, C, N)

        # (B, C)
        denominator = torch.sum(input, dim=-1) + torch.sum(target, dim=-1)
        mask = denominator > 0.
        numerator = 2. * torch.sum(torch.mul(input, target), dim=-1)

        res = torch.ones_like(denominator)
        res[mask] = torch.div(numerator[mask], denominator[mask])
        res = 1. - res

        return self.agg_fun(res)
