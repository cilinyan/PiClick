# Copyright (c) OpenMMLab. All rights reserved.
import torch
import pdb
from scipy.optimize import linear_sum_assignment
from abc import ABCMeta, abstractmethod
from .match_cost import ClassificationCost, FocalLossCost, DiceCost, DistCost
from .assign_result import AssignResult


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""


class MaskHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth for
    mask.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, mask focal cost and mask dice cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost (:obj:`mmcv.ConfigDict` | dict): Classification cost config.
        mask_cost (:obj:`mmcv.ConfigDict` | dict): Mask cost config.
        dice_cost (:obj:`mmcv.ConfigDict` | dict): Dice cost config.
    """

    def __init__(self,
                 cls_cost: dict = dict(type='ClassificationCost', weight=1.0),
                 mask_cost: dict = dict(type='FocalLossCost', weight=1.0, binary_input=True),
                 dice_cost: dict = dict(type='DiceCost', weight=1.0),
                 **kwargs):
        if cls_cost['type'] == 'ClassificationCost':
            self.cls_cost = ClassificationCost(**cls_cost)
        elif cls_cost['type'] == 'DistCost':
            self.cls_cost = DistCost(**cls_cost)
        else:
            raise

        if mask_cost['type'] == 'FocalLossCost':
            self.mask_cost = FocalLossCost(**mask_cost)
        elif mask_cost['type'] == 'CrossEntropyLossCost':
            self.mask_cost = ClassificationCost(**mask_cost)
        else:
            raise

        assert dice_cost['type'] == 'DiceCost'
        self.dice_cost = DiceCost(**dice_cost)

    def assign(self,
               cls_pred,
               mask_pred,
               gt_labels,
               gt_mask,
               gt_bboxes_ignore=None,
               eps=1e-7,
               **kwargs):
        """Computes one-to-one matching based on the weighted costs.

        Args:
            cls_pred (Tensor | None): Class prediction in shape
                (num_query, cls_out_channels).
            mask_pred (Tensor): Mask prediction in shape (num_query, H, W).
            gt_labels (Tensor): Label of 'gt_mask'in shape = (num_gt, ).
            gt_mask (Tensor): Ground truth mask in shape = (num_gt, H, W).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        # K-Net sometimes passes cls_pred=None to this assigner.
        # So we should use the shape of mask_pred
        num_gt, num_query = gt_labels.shape[0], mask_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = mask_pred.new_full((num_query,), -1, dtype=torch.long)
        assigned_labels = mask_pred.new_full((num_query,), -1, dtype=torch.long)
        if num_gt == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            if num_gt == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and maskcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0

        if self.mask_cost.weight != 0:
            # mask_pred shape = [num_query, h, w]
            # gt_mask shape = [num_gt, h, w]
            # mask_cost shape = [num_query, num_gt]
            mask_cost = self.mask_cost(mask_pred, gt_mask)
        else:
            mask_cost = 0

        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(mask_pred, gt_mask)
        else:
            dice_cost = 0
        cost = cls_cost + mask_cost + dice_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(mask_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(mask_pred.device)
        # pdb.set_trace()
        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

    def assign_match(self,
                     cls_pred,
                     mask_pred,
                     gt_labels,
                     gt_mask,
                     gt_bboxes_ignore=None,
                     eps=1e-7,
                     **kwargs):
        """Computes one-to-one matching based on the weighted costs.

        Args:
            cls_pred (Tensor | None): Class prediction in shape
                (num_query, cls_out_channels).
            mask_pred (Tensor): Mask prediction in shape (num_query, H, W).
            gt_labels (Tensor): Label of 'gt_mask'in shape = (num_gt, ).
            gt_mask (Tensor): Ground truth mask in shape = (num_gt, H, W).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        # K-Net sometimes passes cls_pred=None to this assigner.
        # So we should use the shape of mask_pred
        num_gt, num_query = gt_labels.shape[0], mask_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = mask_pred.new_full((num_query,), -1, dtype=torch.long)
        assigned_labels = mask_pred.new_full((num_query,), -1, dtype=torch.long)
        if num_gt == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            if num_gt == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and maskcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0

        if self.mask_cost.weight != 0:
            # mask_pred shape = [num_query, h, w]
            # gt_mask shape = [num_gt, h, w]
            # mask_cost shape = [num_query, num_gt]
            mask_cost = self.mask_cost(mask_pred, gt_mask)
        else:
            mask_cost = 0

        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(mask_pred, gt_mask)
        else:
            dice_cost = 0
        cost = cls_cost + mask_cost + dice_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(mask_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(mask_pred.device)
        # pdb.set_trace()
        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds
        return assigned_gt_inds
