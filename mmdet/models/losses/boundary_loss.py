# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..builder import LOSSES


def weighted_bce(bd_pre, target, label):
    num_rois = bd_pre.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=bd_pre.device)
    bd_pre = bd_pre[inds, label].squeeze(1)

    log_p = bd_pre.view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


@LOSSES.register_module()
class BoundaryLoss(nn.Module):
    """Boundary loss.

    This function is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L122>`_.  # noqa
    Licensed under the MIT License.


    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_boundary'):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

    def forward(self, bd_pre: Tensor, bd_gt: Tensor, label) -> Tensor:
        """Forward function.
        Args:
            bd_pre (Tensor): Predictions of the boundary head.
            bd_gt (Tensor): Ground truth of the boundary.

        Returns:
            Tensor: Loss tensor.
        """
        return self.loss_weight * weighted_bce(bd_pre, bd_gt, label)

    @property
    def loss_name(self):
        return self.loss_name_