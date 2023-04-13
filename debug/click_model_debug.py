# PYTHONPATH=".":$PYTHONPATH python debug/click_model_debug.py
import pdb
import torch
from isegm.model.is_multimask_with_ranking_model_debug import MultiMaskWithRankingModel

_PARAMS = dict(
    num_queries=7,
    num_classes=1,
)

TRAIN_CFG: dict = dict(
    assigner=dict(type='MaskHungarianAssigner',
                  cls_cost=dict(type='ClassificationCost', weight=1.0),
                  mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
                  dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
    sampler=dict(type='MaskPseudoSampler')
)

_DEVICE = torch.device('cuda')


def get_model():
    backbone_params = dict(
        img_size=(448, 448),
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
        num_transformer_feat_level=3,
    )

    model = MultiMaskWithRankingModel(
        num_classes=_PARAMS['num_classes'],
        num_queries=_PARAMS['num_queries'],
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        head_params=head_params,
        neck_params=neck_params,
    )

    model.to(_DEVICE)

    return model


def main():
    model = get_model()
    image = torch.randn((2, 4, 448, 448), dtype=torch.float32, device=_DEVICE)
    points = torch.tensor([
        [[1, 1, 0], [-1, -1, -1], ] * 6 + [[-1, -1, -1], [100, 454, 1], ] * 6,
        [[1, 1, 0], [2, 20, 100], ] * 6 + [[-1, -1, -1], [100, 222, 1], ] * 6,
    ], dtype=torch.long, device=_DEVICE)
    out = model(image, points)
    pdb.set_trace()
    pass


if __name__ == '__main__':
    main()
