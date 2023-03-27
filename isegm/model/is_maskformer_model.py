import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.maskformer_helper.pixel_decoder import PixelDecoder
from .modeling.maskformer_helper.transformer import DetrTransformerDecoder
from .modeling.maskformer_helper.positional_encoding import SinePositionalEncoding

from mmcv.cnn import Conv2d, caffe2_xavier_init

_PARAMS = dict(
    num_queries=7,
    num_classes=1,
)
_BACKBONE_PARAMS = dict(
    img_size=(224, 224),
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
    transformer_decoder_num_layers=6,
)


def select_max_score_mask(cls_scores_list, mask_preds_list, batch_first: bool = False):
    if batch_first:
        cls_scores_list = cls_scores_list.transpose(0, 1)
        mask_preds_list = mask_preds_list.transpose(0, 1)
    cls_scores_list = cls_scores_list[-1]
    mask_preds_list = mask_preds_list[-1]
    indexes = torch.argmax(cls_scores_list.softmax(dim=-1)[:, :, 0], dim=-1)
    max_scores_masks = list()
    for i, m in zip(indexes, mask_preds_list):
        max_scores_masks.append(m[i:i + 1])
    max_scores_masks = torch.stack(max_scores_masks)
    return max_scores_masks


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


class MaskFormerModel(ISModel):
    @serialize
    def __init__(
            self,
            num_classes: int = _PARAMS['num_classes'],
            num_queries: int = _PARAMS['num_queries'],
            backbone_params: dict = _BACKBONE_PARAMS,
            neck_params: dict = _NECK_PARAMS,
            head_params: dict = _HEAD_PARAMS,
            random_split=False,
            **kwargs
    ):
        self.num_classes = num_classes
        self.num_queries = num_queries
        super().__init__(**kwargs)
        self.random_split = random_split

        self.patch_embed_coords = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=3 if self.with_prev_mask else 2,
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = MaskFormerHead(**head_params)

    def backbone_forward(self, image, coord_features=None):
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features, self.random_split)
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size
        backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        multi_scale_features = self.neck(backbone_features)

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


class MaskFormerModelV2(ISModel):
    @serialize
    def __init__(
            self,
            num_classes: int = _PARAMS['num_classes'],
            num_queries: int = _PARAMS['num_queries'],
            backbone_params: dict = _BACKBONE_PARAMS,
            head_params: dict = _HEAD_PARAMS,
            random_split=False,
            num_scale: int = 4,
            **kwargs
    ):

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_scale = num_scale
        super().__init__(**kwargs)
        self.random_split = random_split

        self.patch_embed_coords = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=3 if self.with_prev_mask else 2,
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)

        head_params['in_channels'] = [backbone_params['embed_dim']] * self.num_scale

        self.head = MaskFormerHead(**head_params)

    def backbone_forward(self, image, coord_features=None):
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = \
            self.backbone.forward_multi_scale(image, coord_features, self.random_split, num_stage=self.num_scale)
        # B, C, grid_size
        feature_shape = [(f.shape[0], f.shape[2], round(f.shape[1] ** 0.5)) for f in backbone_features]
        multi_scale_features = \
            [f.transpose(-1, -2).view(b, c, gs, gs) for f, (b, c, gs) in zip(backbone_features, feature_shape)]

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
        batch_size, C, input_img_h, input_img_w = feats[0].shape
        img_h, img_w = input_img_h, input_img_w
        padding_mask = feats[-1].new_ones((batch_size, input_img_h, input_img_w), dtype=torch.float32)
        for i in range(batch_size):
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(padding_mask.unsqueeze(1),
                                     size=feats[-1].shape[-2:],
                                     mode='nearest').to(torch.bool).squeeze(1)
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
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
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
