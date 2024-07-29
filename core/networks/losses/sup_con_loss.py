#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import torch.nn as nn
import torch
from typing import Literal


class SupConLoss(nn.Module):
    """
    Gu's implementation of supervised contrastive learning[https://arxiv.org/abs/2004.11362]
    (which also supports unsupervised learning), using the fashion of
    query and key.
    This implementation heavily borrows from the author's
    implementation: https://github.com/HobbitLong/SupContrast/blob/master/losses.py

    This implementation ues PyTorch CrossEntropy
    function, which is more stable and faster than a manual calculation
    of negative log-likelihood and softmax in the original
    implementation.
    """

    def __init__(
            self,
            temperature=0.07,
            contrast_mode: Literal['all', 'one'] = 'all',
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.contrast_mode = contrast_mode

    def forward(
            self,
            query: torch.Tensor,
            key:torch.Tensor,
            labels=None,
    ) -> torch.Tensor:
        # query: (B, C), key: (B, D), labels: (B, )

        B, C = query.shape
        feats = torch.cat([query, key], dim=0)  # (2 * B, C)
        device = feats.device
        dtype = feats.dtype

        pair_label = torch.arange(B, dtype=torch.long, device=device)

        if self.contrast_mode == 'all':
            # (2 * B, 2 * B)
            logits = torch.matmul(feats, feats.T)
            n_queryies = 2
            pair_label = torch.cat(
                [pair_label + B, pair_label], dim=0)
            """
                For a case of B=2, pair_label = [2, 3, 0, 1]
                positive_mask = [
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ]
            """
        elif self.contrast_mode == 'one':
            # (B, 2 * B)
            logits = torch.matmul(query, feats.T)
            n_queryies = 1
        else:
            raise NotImplementedError
        logits = torch.div(logits, self.temperature)

        if labels is None:
            # (B * B)
            positive_mask = torch.eye(B, device=device, dtype=dtype)
        else:
            labels = labels.view(-1, 1).contiguous()
            positive_mask = torch.eq(
                labels, labels.T).to(dtype).to(device)
            # (2 * B, 2 * B)
        positive_mask = positive_mask.repeat(n_queryies, 2)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(positive_mask),
            1,
            torch.arange(B * n_queryies).view(-1, 1).to(device),
            0
        )
        positive_mask = positive_mask * logits_mask
        # (2 * B, 2 * B)
        logits = logits * logits_mask

        # (2 * B). For un supervised learning, this should be all 1
        mask_pos_pairs = positive_mask.sum(1)

        # This edge handling below is from the original author.
        # Modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = torch.where(
            mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        loss = self.criterion(logits, pair_label)
        loss = torch.div(loss, mask_pos_pairs)
        return loss.mean()