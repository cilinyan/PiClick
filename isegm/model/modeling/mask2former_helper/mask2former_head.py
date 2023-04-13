# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmcv.runner import ModuleList
from .msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from ..maskformer_helper.positional_encoding import SinePositionalEncoding
from ..maskformer_helper.transformer import DetrTransformerDecoderLayer
from .sub_module import CrossAttention
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence

from easydict import EasyDict as edict

_PIXEL_DECODER_CFG = edict(dict(
    type='MSDeformAttnPixelDecoder',
    num_outs=3,
    norm_cfg=dict(type='GN', num_groups=32),
    act_cfg=dict(type='ReLU'),
    encoder=dict(
        type='DetrTransformerEncoder',
        num_layers=6,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=dict(
                type='MultiScaleDeformableAttention',
                embed_dims=256,
                num_heads=8,
                num_levels=3,
                num_points=4,
                im2col_step=64,
                dropout=0.0,
                batch_first=False,
                norm_cfg=None,
                init_cfg=None),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type='ReLU', inplace=True)),
            operation_order=('self_attn', 'norm', 'ffn', 'norm')),
        init_cfg=None),
    positional_encoding=dict(
        type='SinePositionalEncoding', num_feats=128, normalize=True),
    init_cfg=None
))

_RANKING_CFG = edict(dict(
    type='DetrTransformerDecoderLayer',
    attn_cfgs=dict(
        type='MultiheadAttention',
        embed_dims=256,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=None,
        batch_first=False),
    ffn_cfgs=dict(
        embed_dims=256,
        feedforward_channels=2048,
        num_fcs=2,
        act_cfg=dict(type='ReLU', inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True),
    feedforward_channels=2048,
    operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')
))

_POSITIONAL_ENCODING_CFG = edict(dict(type='SinePositionalEncoding', num_feats=128, normalize=True))

_TRANSFORMER_DECODER_CFG = edict(dict(
    type='DetrTransformerDecoder',
    return_intermediate=True,
    num_layers=9,
    transformerlayers=dict(
        type='DetrTransformerDecoderLayer',
        attn_cfgs=dict(
            type='MultiheadAttention',
            embed_dims=256,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            dropout_layer=None,
            batch_first=False),
        ffn_cfgs=dict(
            embed_dims=256,
            feedforward_channels=2048,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.0,
            dropout_layer=None,
            add_identity=True),
        feedforward_channels=2048,
        operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')),
    init_cfg=None
))


class Mask2FormerHead(nn.Module):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_classes (int): Number of classes
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 feat_channels=256,
                 out_channels=256,
                 num_classes=1,
                 num_queries=7,
                 num_transformer_feat_level=3,
                 pixel_decoder=_PIXEL_DECODER_CFG,
                 enforce_decoder_input_project=False,
                 transformer_decoder=_TRANSFORMER_DECODER_CFG,
                 positional_encoding=_POSITIONAL_ENCODING_CFG,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(in_channels=in_channels, feat_channels=feat_channels, out_channels=out_channels)
        assert pixel_decoder_['type'] == 'MSDeformAttnPixelDecoder'
        self.pixel_decoder = MSDeformAttnPixelDecoder(**pixel_decoder_)
        # self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if self.decoder_embed_dims != feat_channels or enforce_decoder_input_project:
                self.decoder_input_projs.append(Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        assert positional_encoding['type'] == 'SinePositionalEncoding'
        self.decoder_positional_encoding = SinePositionalEncoding(**positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.init_weights()

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h*w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(self, feats):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_size = feats[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:]
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return torch.stack(cls_pred_list), torch.stack(mask_pred_list)


class ClickAddMask2FormerHead(nn.Module):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_classes (int): Number of classes
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 feat_channels=256,
                 out_channels=256,
                 num_classes=1,
                 num_queries=7,
                 num_transformer_feat_level=3,
                 pixel_decoder=_PIXEL_DECODER_CFG,
                 enforce_decoder_input_project=False,
                 transformer_decoder=_TRANSFORMER_DECODER_CFG,
                 positional_encoding=_POSITIONAL_ENCODING_CFG,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(in_channels=in_channels, feat_channels=feat_channels, out_channels=out_channels)
        assert pixel_decoder_['type'] == 'MSDeformAttnPixelDecoder'
        self.pixel_decoder = MSDeformAttnPixelDecoder(**pixel_decoder_)
        # self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if self.decoder_embed_dims != feat_channels or enforce_decoder_input_project:
                self.decoder_input_projs.append(Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        assert positional_encoding['type'] == 'SinePositionalEncoding'
        self.decoder_positional_encoding = SinePositionalEncoding(**positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.ranking_module = DetrTransformerDecoderLayer(**_RANKING_CFG)
        self.iou_embed = nn.Linear(feat_channels, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward_head(self, query_mask, mask_feature, attn_mask_target_size, query_points, points_attn_mask):
        """Forward for head part which is called after every decoder layer.

        Args:
            query_mask (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention mask size.
            query_points (Tensor): [num_points, batch_size, c]
            points_attn_mask (Tensor): [batch_size, num_points]

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
            - iou_pred (Tensor): IoU score in shape (batch_size, num_queries, 1)
        """
        num_masks, batch_size, _ = query_mask.shape
        decoder_in = torch.cat([query_mask, query_points], dim=0)
        decoder_out = self.transformer_decoder.post_norm(decoder_in)
        query_mask, query_points = decoder_out[:num_masks], decoder_out[num_masks:]

        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        # import pdb; pdb.set_trace()
        # [batch_size, num_points] -> [batch_size, num_heads, num_queries, num_points] -> [batch_size*num_heads, num_queries, num_points]
        rank_attn_mask = \
            points_attn_mask[:, None, None, :].repeat((1, self.num_heads, self.num_queries, 1)).flatten(0, 1)

        # ranking module
        iou_embed = self.ranking_module(
            query=query_mask,  # (num_queries, batch_size, c)
            key=query_points,
            value=query_points,
            query_pos=query_embed,  # 这个加的应该没问题吧 ?
            key_pos=None,
            attn_masks=[rank_attn_mask, None],  # 这个加的应该没问题吧 ?
            query_key_padding_mask=None,
            key_padding_mask=None,
        )
        iou_pred = self.iou_embed(iou_embed.transpose(0, 1))  # batch, num_queries, 1

        query_mask = query_mask.transpose(0, 1)
        # shape (batch_size, num_queries, c)

        cls_pred = self.cls_embed(query_mask)
        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(query_mask)
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
        # shape (batch_size, num_queries, h, w) -> (batch_size * num_head, num_queries, h*w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask, iou_pred

    def forward(self, feats, query_points, points_attn_mask):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the upstream network, each is a 4D-tensor.
            query_points (Tensor): [num_points, batch_size, c]

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, h, w).
        """
        num_points, _, query_dim = query_points.shape
        batch_size = feats[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        iou_pred_list = []
        cls_pred, mask_pred, attn_mask, iou_pred = self.forward_head(query_feat,
                                                                     mask_features,
                                                                     multi_scale_memorys[0].shape[-2:],
                                                                     query_points,
                                                                     points_attn_mask)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        iou_pred_list.append(iou_pred)

        for i in range(self.num_transformer_decoder_layers):
            point_query_embed = torch.zeros((num_points, batch_size, query_dim)).to(query_embed)
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]

            # attn_mask: batch_size * num_head, num_queries, h*w
            bn, _, hw = attn_mask.shape
            # 不 mask 任何 points 的 query
            attn_mask_points = torch.zeros((bn, num_points, hw), dtype=torch.bool).to(attn_mask)
            attn_masks = [torch.cat([attn_mask, attn_mask_points], dim=1), None]

            query_out = layer(
                query=torch.cat([query_feat, query_points], dim=0),
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=torch.cat([query_embed, point_query_embed], dim=0),
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            query_feat, query_points = query_out[:self.num_queries], query_out[self.num_queries:]
            cls_pred, mask_pred, attn_mask, iou_pred = self.forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                query_points,
                points_attn_mask
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            iou_pred_list.append(iou_pred)

        return torch.stack(cls_pred_list), torch.stack(mask_pred_list), torch.stack(iou_pred_list)


class ClickAddMaskSimple2FormerHead(nn.Module):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_classes (int): Number of classes
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 feat_channels=256,
                 out_channels=256,
                 num_classes=1,
                 num_queries=7,
                 num_transformer_feat_level=3,
                 pixel_decoder=_PIXEL_DECODER_CFG,
                 enforce_decoder_input_project=False,
                 transformer_decoder=_TRANSFORMER_DECODER_CFG,
                 positional_encoding=_POSITIONAL_ENCODING_CFG,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(in_channels=in_channels, feat_channels=feat_channels, out_channels=out_channels)
        assert pixel_decoder_['type'] == 'MSDeformAttnPixelDecoder'
        self.pixel_decoder = MSDeformAttnPixelDecoder(**pixel_decoder_)
        # self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if self.decoder_embed_dims != feat_channels or enforce_decoder_input_project:
                self.decoder_input_projs.append(Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        assert positional_encoding['type'] == 'SinePositionalEncoding'
        self.decoder_positional_encoding = SinePositionalEncoding(**positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.iou_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, 1))

        self.init_weights()

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward_head(self, query_mask, mask_feature, attn_mask_target_size, query_points, points_attn_mask):
        """Forward for head part which is called after every decoder layer.

        Args:
            query_mask (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention mask size.
            query_points (Tensor): [num_points, batch_size, c]
            points_attn_mask (Tensor): [batch_size, num_points]

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
            - iou_pred (Tensor): IoU score in shape (batch_size, num_queries, 1)
        """
        num_masks, batch_size, _ = query_mask.shape
        decoder_in = torch.cat([query_mask, query_points], dim=0)
        decoder_out = self.transformer_decoder.post_norm(decoder_in)
        query_mask, query_points = decoder_out[:num_masks], decoder_out[num_masks:]

        query_mask = query_mask.transpose(0, 1)
        # shape (batch_size, num_queries, c)

        iou_pred = self.iou_embed(query_mask)
        cls_pred = self.cls_embed(query_mask)
        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(query_mask)
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
        # shape (batch_size, num_queries, h, w) -> (batch_size * num_head, num_queries, h*w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask, iou_pred

    def forward(self, feats, query_points, points_attn_mask):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the upstream network, each is a 4D-tensor.
            query_points (Tensor): [num_points, batch_size, c]

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, h, w).
        """
        num_points, _, query_dim = query_points.shape
        batch_size = feats[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        iou_pred_list = []
        cls_pred, mask_pred, attn_mask, iou_pred = self.forward_head(query_feat,
                                                                     mask_features,
                                                                     multi_scale_memorys[0].shape[-2:],
                                                                     query_points,
                                                                     points_attn_mask)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        iou_pred_list.append(iou_pred)

        for i in range(self.num_transformer_decoder_layers):
            point_query_embed = torch.zeros((num_points, batch_size, query_dim)).to(query_embed)
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]

            # attn_mask: batch_size * num_head, num_queries, h*w
            bn, _, hw = attn_mask.shape
            # 不 mask 任何 points 的 query
            attn_mask_points = torch.zeros((bn, num_points, hw), dtype=torch.bool).to(attn_mask)
            attn_masks = [torch.cat([attn_mask, attn_mask_points], dim=1), None]

            query_out = layer(
                query=torch.cat([query_feat, query_points], dim=0),
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=torch.cat([query_embed, point_query_embed], dim=0),
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            query_feat, query_points = query_out[:self.num_queries], query_out[self.num_queries:]
            cls_pred, mask_pred, attn_mask, iou_pred = self.forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                query_points,
                points_attn_mask
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            iou_pred_list.append(iou_pred)

        return torch.stack(cls_pred_list), torch.stack(mask_pred_list), torch.stack(iou_pred_list)
