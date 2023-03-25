import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(1, '.')

from isegm.utils.exp_imports.default import *
from isegm.model.losses import DETRLikeLoss
from tools.visual import draw_sample, draw_masks, PALETTE
from torch.utils.data import DataLoader
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from isegm.model.is_maskformer_model import MaskFormerModel
from isegm.model.modeling.maskformer_helper.misc import multi_apply

from collections import defaultdict
from loguru import logger
import pdb
from copy import deepcopy
from typing import Union

MODEL_NAME = 'cocolvis_plainvit_base224'

_PARAMS = dict(
    num_queries=7,
    num_classes=1,
)

device = torch.device('cuda')

TRAIN_CFG: dict = dict(
    assigner=dict(type='MaskHungarianAssigner',
                  cls_cost=dict(type='ClassificationCost', weight=1.0),
                  mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
                  dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
    sampler=dict(type='MaskPseudoSampler')
)
assigner = MaskHungarianAssigner(**TRAIN_CFG['assigner'])
sampler = MaskPseudoSampler()


def get_targets(cls_scores_list, mask_preds_list, gt_labels_list,
                gt_masks_list, img_metas):
    labels_list, label_weights_list, mask_targets_list, mask_weights_list, pos_inds_list, neg_inds_list = \
        multi_apply(_get_target_single, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas)

    num_total_pos = sum((inds.numel() for inds in pos_inds_list))
    num_total_neg = sum((inds.numel() for inds in neg_inds_list))
    return labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg


def _get_target_single(cls_score, mask_pred, gt_labels, gt_masks, img_metas):
    target_shape = mask_pred.shape[-2:]
    if gt_masks.shape[0] > 0:
        gt_masks_downsampled = nn.functional.interpolate(gt_masks.unsqueeze(1).float(),
                                                         target_shape, mode='nearest').squeeze(1).long()
    else:
        gt_masks_downsampled = gt_masks

    # assign and sample
    assign_result = assigner.assign(cls_score, mask_pred, gt_labels, gt_masks_downsampled, img_metas)
    sampling_result = sampler.sample(assign_result, mask_pred, gt_masks)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds

    # label target
    labels = gt_labels.new_full((_PARAMS['num_queries'],), _PARAMS['num_classes'], dtype=torch.long)
    labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
    label_weights = gt_labels.new_ones(_PARAMS['num_queries'])

    # mask target
    mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
    mask_weights = mask_pred.new_zeros((_PARAMS['num_queries'],))
    mask_weights[pos_inds] = 1.0

    return labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds


def draw_sample_split(image: torch.Tensor,
                      points: torch.Tensor,
                      mask: Union[torch.Tensor, np.ndarray],
                      data_info: dict,
                      out_path: str = 'debug/vis/points.jpg',
                      ):
    points = points.cpu().numpy()
    img = np.array(image.permute((1, 2, 0)).cpu().numpy() * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = np.array(np.array(mask, dtype=int) == 1)

    img = draw_masks(img, mask, np.array(list(reversed(PALETTE)), dtype=np.uint8), alpha=0.7)
    points_pos, points_neg = points.reshape((2, -1, 3)).astype(int)
    for y, x, tag in points_pos:  # red
        if tag == -1: continue
        img = cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
    for y, x, tag in points_neg:  # blue
        if tag == -1: continue
        img = cv2.circle(img, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
    h, w = data_info['ori_shape']
    img = cv2.resize(img, (w, h))
    if out_path is not None:
        cv2.imwrite(out_path, img)


def get_masks_by_points(points, data_info) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points_pos, points_neg = points.reshape((2, -1, 3)).astype(int)
    layers = data_info['mask']
    gt_mask = list()
    for obj_id, info in data_info['object'].items():
        layer_id, mask_id = info['mapping']
        mask = np.array(layers[:, :, layer_id] == mask_id)
        flag_pos = all((f == -1) or mask[x, y] for x, y, f in points_pos)
        flag_neg = all((f == -1) or (not mask[x, y]) for x, y, f in points_neg)
        if flag_pos and flag_neg:
            gt_mask.append(np.array(deepcopy(mask), dtype=float))
    return np.array(gt_mask)


def collate_fn(values):
    res = defaultdict(list)
    for value in values:
        for k, v in value.items():
            res[k].append(v)
    res = {
        'images': torch.stack(res['images']),
        'points': torch.tensor(np.array(res['points'])),
        'instances': torch.tensor(np.array(res['instances'])),
        'data_info': res['data_info'],
    }
    return res


def main():
    model, model_cfg = init_model()
    train(model, model_cfg)


def init_model():
    model_cfg = edict()
    model_cfg.crop_size = (224, 224)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=(224, 224),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim=768,
        out_dims=[256, 256, 256, 256],
    )

    head_params = dict(
        num_classes=_PARAMS['num_classes'],
        num_queries=_PARAMS['num_queries'],
        in_channels=[256, 256, 256, 256],
        feat_channels=256,
        out_channels=256,
        transformer_decoder_num_layers=6,
    )

    model = MaskFormerModel(
        num_classes=_PARAMS['num_classes'],
        num_queries=_PARAMS['num_queries'],
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
    )

    model.backbone.init_weights_from_pretrained("./weights/mae_pretrain_vit_base.pth")
    model.to(device)

    return model, model_cfg


@logger.catch()
def train(model, model_cfg):
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = DETRLikeLoss(num_queries=_PARAMS['num_queries'], num_classes=_PARAMS['num_classes'])
    loss_cfg.instance_loss_weight = 1.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = CocoLvisDataset(
        "./datasets/LVIS",
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.30
    )

    valset = CocoLvisDataset(
        "./datasets/LVIS",
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )

    train_data = DataLoader(
        trainset, 32,
        sampler=get_sampler(trainset, shuffle=True, distributed=False),
        drop_last=True, pin_memory=True,
        num_workers=8,
        collate_fn=collate_fn,
    )

    # import pdb; pdb.set_trace()
    for batch_data in train_data:
        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
        image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
        gt_masks = [get_masks_by_points(p, i) for p, i in zip(batch_data['points'], batch_data['data_info'])]
        orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()
        prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
        net_input = torch.cat((image, prev_output), dim=1)
        output = model(net_input, points)
        draw_sample_split(image[0], points[0], gt_mask[0], batch_data['data_info'][0])

        gt_labels = [torch.tensor([0] * m.shape[0], dtype=torch.int).to(device).long() for m in gt_masks]
        gt_masks = [torch.tensor(g).to(device) for g in gt_masks]
        cls_scores_list, mask_preds_list = output['instances']
        img_metas = [None for _ in gt_masks]
        # [g.shape for g in gt_masks]
        # i = 22; draw_sample_split(image[i], points[i], gt_mask[i], batch_data['data_info'][i])
        # i = 1; draw_sample_split(image[i], points[i], gt_masks[i], batch_data['data_info'][i])
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg = \
            get_targets(cls_scores_list[-1], mask_preds_list[-1], gt_labels, gt_masks, img_metas)
        pdb.set_trace()


if __name__ == '__main__':
    main()
