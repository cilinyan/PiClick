import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from mmcv.cnn import Conv2d, caffe2_xavier_init

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

_HEAD_PARAMS = dict(
    num_classes=_PARAMS['num_classes'],
    num_queries=_PARAMS['num_queries'],
    out_channels=256,
    transformer_decoder_num_layers=6,
)
TRAIN_CFG: dict = dict(
    assigner=dict(type='MaskHungarianAssigner',
                  cls_cost=dict(type='ClassificationCost', weight=1.0),
                  mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
                  dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
    sampler=dict(type='MaskPseudoSampler')
)


def select_max_score_mask(cls_scores_list, mask_preds_list, batch_first: bool = False):
    if batch_first:
        cls_scores_list = cls_scores_list[:, -1, ...]
        mask_preds_list = mask_preds_list[:, -1, ...]
    else:
        cls_scores_list = cls_scores_list[-1]
        mask_preds_list = mask_preds_list[-1]
    indexes = torch.argmax(cls_scores_list.softmax(dim=-1)[:, :, 0], dim=-1)
    indexes = indexes.reshape(-1, 1, 1, 1).repeat(1, 1, *mask_preds_list.shape[-2:])
    max_scores_masks = torch.gather(mask_preds_list, 1, indexes)
    return max_scores_masks


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
    return torch.stack(masks_choice)


class MultiMaskBackboneMultiStageModel(ISModel):
    @serialize
    def __init__(
            self,
            num_classes: int = _PARAMS['num_classes'],
            num_queries: int = _PARAMS['num_queries'],
            backbone_params: dict = _BACKBONE_PARAMS,
            head_params: dict = _HEAD_PARAMS,
            backbone_output_blocks=(4, 6, 8, 10, 12),
            multi_scale_feature_dim=512,
            **kwargs
    ):
        self.num_classes = num_classes
        self.num_queries = num_queries
        super().__init__(**kwargs)

        self.patch_embed_coords = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=3 if self.with_prev_mask else 2,
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone_output_blocks = backbone_output_blocks
        self.backbone_embed_dim = backbone_params['embed_dim']
        self.backbone = VisionTransformer(**backbone_params)

        self.multi_scale_feature_dim = multi_scale_feature_dim
        self.down_4_dict = nn.ModuleDict({
            str(block): nn.Sequential(
                nn.ConvTranspose2d(self.backbone_embed_dim, self.backbone_embed_dim, 2, stride=2),
                nn.GroupNorm(1, self.backbone_embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.backbone_embed_dim, self.backbone_embed_dim // 2, 2, stride=2),
                nn.GroupNorm(1, self.backbone_embed_dim // 2),
                nn.Conv2d(self.backbone_embed_dim // 2, self.multi_scale_feature_dim, 1),
                nn.GroupNorm(1, self.multi_scale_feature_dim),
                nn.GELU()
            )
            for block in self.backbone_output_blocks
        })

        head_params.update(dict(in_channels=[self.multi_scale_feature_dim for _ in self.backbone_output_blocks]))
        self.head = MaskFormerHead(**head_params)

        self.assigner = MaskHungarianAssigner(**TRAIN_CFG['assigner'])
        self.sampler = MaskPseudoSampler()

        del self.backbone.head
        del self.backbone.pre_logits
        del self.backbone.fc_norm

    def backbone_forward(self, image, coord_features=None):
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_multi_scale_original(image,
                                                                       coord_features,
                                                                       self.backbone_output_blocks)
        # B, C, grid_size
        feature_shape = [(f.shape[0], f.shape[2], round(f.shape[1] ** 0.5)) for f in backbone_features]
        multi_scale_features = [f.transpose(-1, -2).view(b, c, gs, gs).contiguous()
                                for f, (b, c, gs) in zip(backbone_features, feature_shape)]
        multi_scale_features = [self.down_4_dict[str(block)](feature)
                                for block, feature in zip(self.backbone_output_blocks, multi_scale_features)]
        assert multi_scale_features[0].shape[-1] * 4 == image.shape[-1], \
            f'multi_scale_features[0].shape: {multi_scale_features[0]}, image.shape: {image.shape}'
        return {'instances': self.head(multi_scale_features), 'instances_aux': None}

    def forward(self, image, points, batch_first=False, **kwargs):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image, coord_features)

        if not isinstance(outputs['instances'], tuple):
            outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        else:
            all_cls_scores, all_mask_preds = outputs['instances']
            h_img, w_img = image.shape[-2:]
            num_layer, batch_size, num_queries, h_feat, w_feat = all_mask_preds.shape
            all_mask_preds = torch.reshape(all_mask_preds, (num_layer * batch_size, num_queries, h_feat, w_feat))
            all_mask_preds = nn.functional.interpolate(all_mask_preds, size=image.size()[2:],
                                                       mode='bilinear', align_corners=True)
            all_mask_preds = torch.reshape(all_mask_preds, (num_layer, batch_size, num_queries, h_img, w_img))
            if batch_first:
                outputs['instances'] = all_cls_scores.transpose(0, 1), all_mask_preds.transpose(0, 1)
            else:
                outputs['instances'] = all_cls_scores, all_mask_preds

        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                                 mode='bilinear', align_corners=True)

        if kwargs.get('test_mode', False):
            outputs['instances'] = select_max_score_mask(*outputs['instances'], batch_first=batch_first)

        if kwargs.get('train_mode', False):
            outputs['single_masks'] = select_max_score_mask(*outputs['instances'], batch_first=batch_first)

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
        labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg = \
            self.mask_match(batch_data, gt_mask, output)
        masks_choice = choice_mask(labels_list, output['instances'][1][-1])
        return masks_choice


class MaskFormerHead(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_queries=7,
                 in_channels=[192, 384, 768, 1536],
                 feat_channels=256,
                 out_channels=256,
                 transformer_decoder_num_layers=6,
                 enforce_decoder_input_project=True,
                 ):
        super(MaskFormerHead, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.pixel_decoder = PixelDecoder(in_channels=in_channels,
                                          feat_channels=feat_channels,
                                          out_channels=out_channels,
                                          norm_cfg=dict(type='GN', num_groups=32),
                                          act_cfg=dict(type='ReLU'))
        self.transformer_decoder = DetrTransformerDecoder(
            return_intermediate=True,
            num_layers=transformer_decoder_num_layers,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.1,
                    dropout_layer=None,
                    add_identity=True),
                # the following parameter was not used, just make current api happy
                feedforward_channels=2048,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None
        )

        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        if self.decoder_embed_dims != in_channels[-1] or enforce_decoder_input_project:
            self.decoder_input_proj = Conv2d(in_channels[-1], self.decoder_embed_dims, kernel_size=1)
        else:
            self.decoder_input_proj = nn.Identity()

        self.decoder_pe = SinePositionalEncoding(num_feats=128, normalize=True)
        self.query_embed = nn.Embedding(self.num_queries, out_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                                        nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                                        nn.Linear(feat_channels, out_channels))

        self.init_weights()

    def init_weights(self):
        if isinstance(self.decoder_input_proj, Conv2d):
            caffe2_xavier_init(self.decoder_input_proj, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats, ):
        B, _, H, W = feats[0].shape
        padding_mask = feats[0].new_zeros((B, H, W), dtype=torch.bool)  # B, H, W
        # when backbone is swin, memory is output of last stage of swin.
        # when backbone is r50, memory is output of tranformer encoder.
        mask_features, memory = self.pixel_decoder(feats)
        pos_embed = self.decoder_pe(padding_mask)
        memory = self.decoder_input_proj(memory)
        # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
        memory = memory.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # shape (batch_size, h * w)
        padding_mask = padding_mask.flatten(1)
        # shape = (num_queries, embed_dims)
        query_embed = self.query_embed.weight
        # shape = (num_queries, batch_size, embed_dims)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
        target = torch.zeros_like(query_embed)
        # shape (num_decoder, num_queries, batch_size, embed_dims)
        out_dec = self.transformer_decoder(query=target,
                                           key=memory,
                                           value=memory,
                                           key_pos=pos_embed,
                                           query_pos=query_embed,
                                           key_padding_mask=padding_mask)
        # shape (num_decoder, batch_size, num_queries, embed_dims)
        out_dec = out_dec.transpose(1, 2)

        # cls_scores
        all_cls_scores = self.cls_embed(out_dec)

        # mask_preds
        mask_embed = self.mask_embed(out_dec)
        all_mask_preds = torch.einsum('lbqc,bchw->lbqhw', mask_embed, mask_features)

        return all_cls_scores, all_mask_preds
