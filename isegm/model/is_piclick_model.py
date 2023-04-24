import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import point_sample
import numpy as np
import random
from copy import deepcopy
from isegm.utils.serialization import serialize
from .modeling.maskformer_helper.pixel_decoder import PixelDecoder
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.maskformer_helper.transformer import DetrTransformerDecoder
from .modeling.maskformer_helper.positional_encoding import SinePositionalEncoding
from .modeling.maskformer_helper.mask_hungarian_assigner import MaskHungarianAssigner
from .modeling.maskformer_helper.mask_pseudo_sampler import MaskPseudoSampler
from .modeling.maskformer_helper.misc import multi_apply
from .modeling.mask2former_helper.piclick_head import PiClickHead

from mmcv.cnn import (build_activation_layer, build_conv_layer, build_norm_layer, xavier_init)
from loguru import logger

_PARAMS = dict(
    num_queries=7,
    num_classes=1,
)
_BACKBONE_PARAMS = dict(
    img_size=(448, 448),
    patch_size=(16, 16),
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
)
_NECK_PARAMS = dict(
    in_dim=768,
    out_dims=[256, 256, 256, 256],
)
_HEAD_PARAMS = dict(
    num_classes=_PARAMS['num_classes'],
    num_queries=_PARAMS['num_queries'],
    in_channels=[256, 256, 256, 256],
    feat_channels=256,
    out_channels=256,
    num_transformer_feat_level=3,
)
TRAIN_CFG: dict = dict(
    assigner=dict(type='MaskHungarianAssigner',
                  cls_cost=dict(type='ClassificationCost', weight=1.0),
                  mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
                  dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
    sampler=dict(type='MaskPseudoSampler')
)


def select_all_possible(cls_scores_list,
                        mask_preds_list,
                        rank_scores_list,
                        rank_multi: float = .8,
                        batch_first: bool = False,
                        cls_thr=0.49):
    # 仅在单图推理中使用
    if batch_first:  # B, NUM_LAYER, NUM_QUERY, NUM_CLASS -> B, NUM_QUERY, NUM_CLASS
        cls_scores_list = cls_scores_list[:, -1, ...]
        mask_preds_list = mask_preds_list[:, -1, ...]
        rank_scores_list: torch.Tensor = rank_scores_list[:, -1, ...]
    else:  # NUM_LAYER, B, NUM_QUERY, NUM_CLASS -> B, NUM_QUERY, NUM_CLASS
        cls_scores_list = cls_scores_list[-1]
        mask_preds_list = mask_preds_list[-1]
        rank_scores_list: torch.Tensor = rank_scores_list[-1]

    cls_scores_list = cls_scores_list.softmax(dim=-1)[:, :, 0]
    rank_scores_list = F.sigmoid(rank_scores_list)[:, :, 0]

    scores = cls_scores_list / (1. + rank_multi) + rank_scores_list * rank_multi / (1. + rank_multi)

    scores = scores[0]  # NUM_QUERY
    mask_preds_list = mask_preds_list[0]  # NUM_QUERY, H, W
    idx_max_score = torch.argmax(scores, dim=-1).item()
    idx_possible = torch.where(scores > cls_thr)[0].cpu().numpy().tolist()
    idx_possible = list(set(idx_possible + [idx_max_score]))
    masks_possible: torch.Tensor = mask_preds_list[idx_possible]  # NUM_SELECT, H, W
    return torch.unsqueeze(masks_possible, dim=0)  # 1, NUM_SELECT, H, W


def select_last_layer(cls_scores, mask_preds, rank_scores, onehot_scores, batch_first: bool = False):
    # mask_preds: NUM_LAYER, B, NUM_QUERY, H, W, torch.Size([10, 2, 7, 448, 448])
    # cls_scores: NUM_LAYER, B, NUM_QUERY, NUM_CLASS, torch.Size([10, 2, 7, 2])
    # rank_scores: NUM_LAYER, B, NUM_QUERY, 1, torch.Size([10, 2, 7, 1])
    if batch_first:  # B, NUM_LAYER, NUM_QUERY, NUM_CLASS -> B, NUM_QUERY, NUM_CLASS
        cls_scores = cls_scores[:, -1, ...]
        mask_preds = mask_preds[:, -1, ...]  # -> B, NUM_QUERY, H, W
        rank_scores: torch.Tensor = rank_scores[:, -1, ...]
        onehot_scores: torch.Tensor = onehot_scores[:, -1, ...]
    else:  # NUM_LAYER, B, NUM_QUERY, NUM_CLASS -> B, NUM_QUERY, NUM_CLASS
        cls_scores = cls_scores[-1]
        mask_preds = mask_preds[-1]  # -> B, NUM_QUERY, H, W
        rank_scores: torch.Tensor = rank_scores[-1]
        onehot_scores: torch.Tensor = onehot_scores[-1]

    cls_scores = cls_scores.softmax(dim=-1)[:, :, 0]  # -> B, NUM_QUERY
    rank_scores = F.sigmoid(rank_scores)[:, :, 0]  # -> B, NUM_QUERY
    onehot_scores = F.sigmoid(onehot_scores)[:, :, 0]  # -> B, NUM_QUERY

    sorted_indices = torch.argsort(rank_scores, dim=1, descending=True)  # -> B, NUM_QUERY
    indices_mask = sorted_indices.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*mask_preds.shape)  # -> B, NUM_QUERY, H, W

    mask_preds = torch.gather(mask_preds, dim=1, index=indices_mask)  # -> B, NUM_QUERY, H, W
    cls_scores = torch.gather(cls_scores, dim=1, index=sorted_indices)  # -> B, NUM_QUERY
    rank_scores = torch.gather(rank_scores, dim=1, index=sorted_indices)  # -> B, NUM_QUERY
    onehot_scores = torch.gather(onehot_scores, dim=1, index=sorted_indices)  # -> B, NUM_QUERY

    return cls_scores, mask_preds, rank_scores, onehot_scores


def select_max_score_mask(cls_scores_list,
                          mask_preds_list,
                          rank_scores_list,
                          onehot_scores_list,
                          rank_multi: float = .8,
                          batch_first: bool = False):
    # (B, NUM_QUERY, NUM_CLASS) && (B, NUM_QUERY, H, W)
    if batch_first:
        cls_scores_list: torch.Tensor = cls_scores_list[:, -1, ...]
        mask_preds_list: torch.Tensor = mask_preds_list[:, -1, ...]
        rank_scores_list: torch.Tensor = rank_scores_list[:, -1, ...]
        onehot_scores_list: torch.Tensor = onehot_scores_list[:, -1, ...]
    else:
        cls_scores_list: torch.Tensor = cls_scores_list[-1]
        mask_preds_list: torch.Tensor = mask_preds_list[-1]
        rank_scores_list: torch.Tensor = rank_scores_list[-1]
        onehot_scores_list: torch.Tensor = onehot_scores_list[-1]

    cls_scores_list = cls_scores_list.softmax(dim=-1)[:, :, 0]
    rank_scores_list = F.sigmoid(rank_scores_list)[:, :, 0]

    scores = cls_scores_list / (1. + rank_multi) + rank_scores_list * rank_multi / (1. + rank_multi)

    indexes = torch.argmax(scores, dim=-1)
    indexes = indexes.reshape(-1, 1, 1, 1).repeat(1, 1, *mask_preds_list.shape[-2:])
    max_scores_masks = torch.gather(mask_preds_list, 1, indexes)
    return max_scores_masks  # B, 1, H, W


def get_masks_by_points(points, data_info, source_mask) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points_pos, points_neg = points.reshape((2, -1, 3)).astype(int)
    layers = data_info['mask']
    gt_mask = list()
    gt_obj_ids = list()
    for obj_id, info in data_info['object'].items():
        layer_id, mask_id = info['mapping']
        mask = np.array(layers[:, :, layer_id] == mask_id)
        flag_pos = all((f == -1) or mask[x, y] for x, y, f in points_pos)
        flag_neg = all((f == -1) or (not mask[x, y]) for x, y, f in points_neg)
        if flag_pos and flag_neg:
            gt_mask.append(np.array(deepcopy(mask), dtype=float))
            gt_obj_ids.append(obj_id)
    if (len(data_info['sample_object_ids']) > 1) or (data_info['sample_object_ids'][0] not in gt_obj_ids):
        if isinstance(source_mask, torch.Tensor):
            source_mask = np.squeeze(source_mask.cpu().numpy(), axis=0)
        gt_mask.append(source_mask)
    return np.array(gt_mask)


def choice_mask(labels_list, mask_preds_list):  # -> B,
    masks_choice = list()
    for labels, mask_preds in zip(labels_list, mask_preds_list):
        labels = labels.cpu().numpy()
        indexes = [i for i, label in enumerate(labels) if label == 0]
        idx = random.choice(indexes)
        masks_choice.append(mask_preds[idx:idx + 1])
    return torch.stack(masks_choice)  # B, 1, H, W


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0] * 2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class PiClickModel(ISModel):
    @serialize
    def __init__(
            self,
            num_classes: int = _PARAMS['num_classes'],
            num_queries: int = _PARAMS['num_queries'],
            backbone_params: dict = _BACKBONE_PARAMS,
            neck_params: dict = _NECK_PARAMS,
            head_params: dict = _HEAD_PARAMS,
            random_split=False,
            click_pos_enc_cfg: dict = dict(type='SinePositionalEncoding', num_feats=128, normalize=True),
            **kwargs
    ):
        self.num_classes = num_classes
        self.num_queries = num_queries
        super().__init__(**kwargs)
        self.random_split = random_split
        self.image_size = backbone_params['img_size']

        self.patch_embed_coords = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=3 if self.with_prev_mask else 2,
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = PiClickHead(**head_params)

        self.assigner = MaskHungarianAssigner(**TRAIN_CFG['assigner'])
        self.sampler = MaskPseudoSampler()

        assert click_pos_enc_cfg['type'] == 'SinePositionalEncoding'
        self.click_positional_encoding = SinePositionalEncoding(**click_pos_enc_cfg)
        self.positive_positional_embed = nn.Embedding(1, click_pos_enc_cfg['num_feats'] * 2)  # [1, num_feats*2]
        self.negative_positional_embed = nn.Embedding(1, click_pos_enc_cfg['num_feats'] * 2)  # [1, num_feats*2]

    @property
    def device(self):
        return next(self.parameters()).device

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def backbone_forward(self, image, query_points, points_attn_mask, coord_features=None):
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features, self.random_split)
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size
        backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1]).contiguous()
        multi_scale_features = self.neck(backbone_features)
        return {'instances': self.head(multi_scale_features, query_points, points_attn_mask), 'instances_aux': None}

    def forward(self, image, points, batch_first=False, **kwargs):
        row, col = image.shape[-2:]
        query_points, points_attn_mask = self.get_batch_click_embedding(points, row, col)
        query_points = query_points.transpose(0, 1).contiguous()  # B, num_points, c -> num_points, B, c

        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)

        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image, query_points, points_attn_mask, coord_features)

        if not isinstance(outputs['instances'], tuple):
            outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        else:
            all_cls_scores, all_mask_preds, all_iou_preds, all_onehot_preds = outputs['instances']
            h_img, w_img = image.shape[-2:]
            num_layer, batch_size, num_queries, h_feat, w_feat = all_mask_preds.shape
            all_mask_preds = torch.reshape(all_mask_preds, (num_layer * batch_size, num_queries, h_feat, w_feat))
            all_mask_preds = nn.functional.interpolate(all_mask_preds, size=image.size()[2:],
                                                       mode='bilinear', align_corners=True)
            all_mask_preds = torch.reshape(all_mask_preds, (num_layer, batch_size, num_queries, h_img, w_img))
            if batch_first:
                outputs['instances'] = (all_cls_scores.transpose(0, 1), all_mask_preds.transpose(0, 1),
                                        all_iou_preds.transpose(0, 1), all_onehot_preds.transpose(0, 1))
            else:
                outputs['instances'] = all_cls_scores, all_mask_preds, all_iou_preds, all_onehot_preds

        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                                 mode='bilinear', align_corners=True)

        if kwargs.get('train_mode', False):
            outputs['single_masks'] = select_max_score_mask(*outputs['instances'], batch_first=batch_first)

        if kwargs.get('last_layer', False):
            outputs['instances'] = select_last_layer(*outputs['instances'], batch_first=batch_first)
            pass

        return outputs

    def mask_match(self, batch_data, gt_mask, output):
        device = output['instances'][1].device
        gt_masks = \
            [get_masks_by_points(p, i, g) for p, i, g in zip(batch_data['points'], batch_data['data_info'], gt_mask)]
        gt_labels = [torch.tensor([0] * m.shape[0], dtype=torch.int).to(device).long() for m in gt_masks]
        gt_masks = [torch.tensor(g).long().to(device) for g in gt_masks]
        cls_scores_list, mask_preds_list = output['instances']
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg = \
            self.get_targets(cls_scores_list[-1], mask_preds_list[-1], gt_labels, gt_masks)
        return labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg

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
        return labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg

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
            img_metas (dict): Image informtation.

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

    def forward_for_iter(self, image, gt_mask, points, batch_data):
        output = self(image, points, batch_first=False)  # output['instances']: NUM_LATER, BS, NC, H, W
        output['instances'] = output['instances'][:2]
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg = \
            self.mask_match(batch_data, gt_mask, output)
        masks_choice = choice_mask(labels_list, output['instances'][1][-1])
        return masks_choice

    def get_single_click_embedding(self, x_in: torch.Tensor, row: int = 448, col: int = 448):
        # x (Tensor): shape [24, 2]
        x = x_in.long()
        # logger.error(f'x, shape: {x.shape}, type: {x.dtype}, min: {x.min()}, max: {x.max()}')

        # Create a mask for points that have not been clicked
        # points_attn_mask = (x[:, 2] == -1)
        points_attn_mask = torch.tensor(
            [not ((flag.item() != -1) and (0 <= x0 < row) and (0 <= x1 < col)) for x0, x1, flag in x],
            dtype=torch.bool,
            device=x.device
        )
        # Create a zero mask with the same size as the image
        zero_mask = torch.zeros((1, row, col), dtype=torch.bool, device=self.device, requires_grad=False)
        # Compute the positional embedding for the clicked points
        click_pos_embed = self.click_positional_encoding(zero_mask)  # [1, num_feats*2, h, w]

        # Split the input tensor into positive and negative points
        n = x.shape[0] // 2
        x_pos, x_neg = x[:n], x[n:]

        # Compute the embedding for the positive points
        x_pos_emb = torch.stack([
            click_pos_embed[0, :, x0, x1] + self.positive_positional_embed.weight[0]
            if (flag.item() != -1) and (0 <= x0 < row) and (0 <= x1 < col)
            else torch.zeros_like(click_pos_embed[0, :, 0, 0])
            for x0, x1, flag in x_pos
        ])  # n, 1, num_feats*2
        # Compute the embedding for the negative points
        x_neg_emb = torch.stack([
            click_pos_embed[0, :, x0, x1] + self.negative_positional_embed.weight[0]
            if (flag.item() != -1) and (0 <= x0 < row) and (0 <= x1 < col)
            else torch.zeros_like(click_pos_embed[0, :, 0, 0])
            for x0, x1, flag in x_neg
        ])  # n, 1, num_feats*2
        # Concatenate the embeddings for positive and negative points
        x_emb = torch.cat([x_pos_emb, x_neg_emb], dim=0)

        # Return the embeddings and the attention mask
        return x_emb, points_attn_mask

    def get_batch_click_embedding(self, x, row, col):
        x_emb, points_attn_mask = list(zip(*[self.get_single_click_embedding(xi, row, col) for xi in x]))
        return torch.stack(x_emb), torch.stack(points_attn_mask)
