"""
python train.py models/iter_mask/maskformer_v2_base224_cocolvis_itermask.py --batch-size=25 --ngpus=1
"""

import sys

sys.path.insert(1, '.')

from isegm.utils.exp_imports.default import *
from isegm.model.losses import DETRLikeLoss
from isegm.model.is_maskformer_model import MaskFormerModelV2

from collections import defaultdict

MODEL_NAME = 'cocolvis_maskformerv2_base224'

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


def main(cfg):
    model, model_cfg = init_model()
    train(model, cfg, model_cfg)


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

    head_params = dict(
        num_classes=_PARAMS['num_classes'],
        num_queries=_PARAMS['num_queries'],
        feat_channels=256,
        out_channels=256,
        transformer_decoder_num_layers=6,
    )

    model = MaskFormerModelV2(
        num_classes=_PARAMS['num_classes'],
        num_queries=_PARAMS['num_queries'],
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        head_params=head_params,
        num_scale=4,
    )

    model.backbone.init_weights_from_pretrained("./weights/mae_pretrain_vit_base.pth")
    model.to(device)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
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

    optimizer_params = {'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8}

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[49, 55], gamma=0.1)

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (49, 1)],
                        image_dump_interval=3000,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3,
                        multi_output=True)
    trainer.run(num_epochs=55, validation=False)
