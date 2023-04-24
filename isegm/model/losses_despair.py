import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch.distributed as dist
from collections import OrderedDict

from isegm.utils import misc
from .modeling.maskformer_helper import (cross_entropy_loss, dice_loss, focal_loss)
from .modeling.maskformer_helper.misc import multi_apply
from .modeling.maskformer_helper.dist_utils import reduce_mean
from .modeling.maskformer_helper.mask_hungarian_assigner import MaskHungarianAssigner
from .modeling.maskformer_helper.mask_pseudo_sampler import MaskPseudoSampler

from loguru import logger
import pickle


def get_iou(predict, gt):
    """Calculate Intersection over Union (IoU) for binary segmentation masks.

    Args:
    - predict (torch.Tensor): Predicted segmentation masks with shape (batch_size, num_preds, H, W) in log space.
    - gt (torch.Tensor): Ground truth segmentation masks with shape (batch_size, 1, H, W).

    Returns:
    - iou (torch.Tensor): IoU scores for each prediction with shape (batch_size, num_preds).
    """
    with torch.no_grad():
        gt = torch.stack(gt).to(predict)
        # logger.info(f'predict: {type(predict)}, gt: {type(gt)}')
        # data = dict(predict=predict.cpu(), gt=gt.cpu())
        # with open('/data/clyan/data.pth', 'wb') as file:
        #     pickle.dump(data, file)
        # raise

        # Convert predicted masks to probability masks
        pred_prob = torch.sigmoid(predict)

        # Threshold the probability masks to create binary masks
        pred_mask = (pred_prob > 0.5).float()

        # Compute the intersection between predicted masks and ground truth masks
        intersection = (pred_mask * gt).sum(dim=(2, 3))

        # Compute the union between predicted masks and ground truth masks
        union = (pred_mask + gt).sum(dim=(2, 3)) - intersection

        # Compute the IoU scores
        iou = intersection / (union + 1e-6)

        # logger.error(f'iou: {iou.dtype}, {iou.shape}')

        return iou


class L1Loss(nn.Module):

    def __init__(self, loss_weight=1., reduction: str = 'mean', **kwargs):
        super().__init__()
        self.weight = loss_weight
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        # x (Tensor): log space, shape [batch_size, num_preds, 1]
        # target (Tensor): probabilities, shape [batch_size, num_preds]
        x = F.sigmoid(x).squeeze(dim=-1)
        loss = F.l1_loss(x, target, reduction=self.reduction)
        return self.weight * loss


def max_to_one(x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `x` of shape [batch_size, num_preds], a (float), b (float),
    for each row xi of x,
    set the maximum value of xi to a, and the rest to b.
    Returns the resulting tensor with the same shape as x.
    """
    # Find the maximum value along the last dimension (num_preds)
    max_values, _ = torch.max(x, dim=-1, keepdim=True)

    # Create a mask of the same shape as x, where 1 corresponds to the maximum value
    mask = torch.eq(x, max_values).type(torch.FloatTensor)

    return mask


def max_min_replace(x, a, b):
    max_val, _ = torch.max(x, dim=1, keepdim=True)
    mask = torch.eq(x, max_val)
    out = torch.where(mask, torch.ones_like(x) * a, torch.ones_like(x) * b)
    return out


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, loss_weight=1., **kwargs):
        super().__init__()
        self.weight = loss_weight

    def forward(self, x: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        # x (Tensor): log space, shape [batch_size, num_preds, 1]
        # target (Tensor): probabilities, shape [batch_size, num_preds]
        x = x.squeeze(dim=-1)
        target = max_min_replace(target, 0.9, 0.1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return self.weight * loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
               / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class DETRLikeDespairLoss(nn.Module):
    def __init__(self,
                 num_queries: int,
                 num_classes: int,
                 loss_cls: dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=[1.0] * 1 + [0.1]),
                 loss_rank: dict = dict(
                     type='L1Loss',
                     loss_weight=0.5, ),
                 loss_onehot: dict = dict(
                     type='SoftTargetCrossEntropy',
                     loss_weight=0.5,
                 ),
                 loss_mask: dict = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=20.0),
                 loss_dice: dict = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     naive_dice=True,
                     loss_weight=1.0),
                 train_cfg: dict = dict(
                     assigner=dict(type='MaskHungarianAssigner',
                                   cls_cost=dict(type='ClassificationCost', weight=1.0),
                                   mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
                                   dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
                     sampler=dict(type='MaskPseudoSampler')
                 ),
                 ):
        super(DETRLikeDespairLoss, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.class_weight = loss_cls.get('class_weight', None)
        assert loss_cls['type'] == 'CrossEntropyLoss'
        self.loss_cls = cross_entropy_loss.CrossEntropyLoss(**loss_cls)
        assert loss_rank['type'] == 'L1Loss'
        self.loss_rank = L1Loss(**loss_rank)
        assert loss_onehot['type'] == 'SoftTargetCrossEntropy'
        self.loss_onehot = SoftTargetCrossEntropy(**loss_onehot)
        assert loss_mask['type'] == 'FocalLoss'
        self.loss_mask = focal_loss.FocalLoss(**loss_mask)
        assert loss_dice['type'] == 'DiceLoss'
        self.loss_dice = dice_loss.DiceLoss(**loss_dice)
        self.train_cfg = train_cfg
        assert train_cfg['assigner']['type'] == 'MaskHungarianAssigner'
        self.assigner = MaskHungarianAssigner(**train_cfg['assigner'])
        assert train_cfg['sampler']['type'] == 'MaskPseudoSampler'
        self.sampler = MaskPseudoSampler(context=self)

    def loss(self,
             all_cls_scores, all_mask_preds, all_rank_scores, all_onehot_scores,
             gt_labels_list, gt_masks_list, single_gt_masks):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            all_rank_scores (Tensor): shape (num_decoder, batch_size, num_queries, 1)
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_single_gt_masks_list = [single_gt_masks for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, loss_rank, loss_onehot = \
            multi_apply(self.loss_single,
                        all_cls_scores, all_mask_preds, all_rank_scores, all_onehot_scores,
                        all_gt_labels_list, all_gt_masks_list, all_gt_single_gt_masks_list, )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_rank'] = loss_rank[-1]
        loss_dict['loss_onehot'] = loss_onehot[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_rank_i, loss_onehot_i \
                in zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], loss_rank[:-1], loss_onehot[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_rank'] = loss_rank_i
            loss_dict[f'd{num_dec_layer}.loss_onehot'] = loss_onehot_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores, mask_preds, rank_scores, onehot_scores,
                    gt_labels_list, gt_masks_list, single_gt_masks):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            rank_scores (Tensor): shape (batch_size, num_queries, 1)
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (n, ). n is the sum of number of stuff
                types and number of instances in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.
            single_gt_masks (Tensor): batch_size, 1, h, w

        Returns:
            tuple[Tensor]: Loss components for outputs from a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        # ranking loss
        bs_iou = get_iou(mask_preds, single_gt_masks)
        loss_rank = self.loss_rank(rank_scores, bs_iou)
        loss_onehot = self.loss_onehot(onehot_scores, bs_iou)

        labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg = \
            self.get_targets(cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        target_shape = mask_targets.shape[-2:]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        # upsample to shape of target
        # shape (num_total_gts, h, w)
        mask_preds = \
            F.interpolate(mask_preds.unsqueeze(1), target_shape, mode='bilinear', align_corners=False).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(mask_preds, mask_targets, avg_factor=num_total_masks)

        # mask loss
        # FocalLoss support input of shape (n, num_class)
        h, w = mask_preds.shape[-2:]
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w, 1)
        mask_preds = mask_preds.reshape(-1, 1)
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w)
        mask_targets = mask_targets.reshape(-1)
        # target is (1 - mask_targets) !!!
        loss_mask = self.loss_mask(mask_preds, 1 - mask_targets, avg_factor=num_total_masks * h * w)

        return loss_cls, loss_mask, loss_dice, loss_rank, loss_onehot

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list):
        """Compute classification and mask targets for all images for a decoder layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, pos_inds_list, neg_inds_list = \
            multi_apply(self._get_target_single, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (n, ). n is the sum of number of stuff type and number
                of instance in a image.
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (n, h, w).

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image.
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image.
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image.
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        target_shape = mask_pred.shape[-2:]
        if gt_masks.shape[0] > 0:
            gt_masks_downsampled = \
                F.interpolate(gt_masks.unsqueeze(1).float(), target_shape, mode='nearest').squeeze(1).long()
        else:
            gt_masks_downsampled = gt_masks

        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_pred, gt_labels, gt_masks_downsampled)
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(self.num_queries)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos_inds] = 1.0

        return labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self,
                outputs: Tuple,
                gt_masks: List[torch.tensor],
                single_gt_masks: List[torch.Tensor],
                ):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        """
        all_cls_scores, all_mask_preds, all_rank_scores, all_onehot_scores = outputs
        device = gt_masks[0].device
        gt_labels = [
            torch.tensor([0] * m.shape[0], dtype=torch.int).long().to(device) for m in gt_masks
        ]
        losses = self.loss(all_cls_scores, all_mask_preds, all_rank_scores, all_onehot_scores,
                           gt_labels, gt_masks, single_gt_masks)
        loss, log_vars = self._parse_losses(losses)
        if torch.distributed.get_rank() == 0:
            logger.warning(str(log_vars))
        return loss
