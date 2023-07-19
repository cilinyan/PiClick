from isegm.utils.exp_imports.default import *
from isegm.model.losses_despair import DETRLikeDespairLoss
from isegm.model.is_piclick_mp_model import PiClickMPModel
from isegm.engine.trainer_piclick import ISTrainerPiClick

"""
python -m torch.distributed.launch --nproc_per_node=7 --master_port=59516 --use_env train.py \
  models/iter_mask/piclick_mp_base448_cocolvis_itermask_3m.py \
  --batch-size=119 \
  --ngpus=7
"""

MODEL_NAME = 'piclick_MP_base448_cocolvis_itermask_3m'

_PARAMS = dict(
    num_queries=3,
    num_classes=1,
)

TRAIN_CFG: dict = dict(
    assigner=dict(type='MaskHungarianAssigner',
                  cls_cost=dict(type='ClassificationCost', weight=1.0),
                  mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
                  dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
    sampler=dict(type='MaskPseudoSampler')
)


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=model_cfg.crop_size,
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

    model = PiClickMPModel(
        num_classes=_PARAMS['num_classes'],
        num_queries=_PARAMS['num_queries'],
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        head_params=head_params,
        neck_params=neck_params,
    )

    model.backbone.init_weights_from_pretrained(cfg.IMAGENET_PRETRAINED_MODELS.MAE_BASE)
    model.to(cfg.device)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = DETRLikeDespairLoss(
        num_queries=_PARAMS['num_queries'],
        num_classes=_PARAMS['num_classes'],
        loss_rank=dict(type='L1Loss', loss_weight=0.3, ),
        loss_onehot=dict(type='SoftTargetCrossEntropy', loss_weight=0.05, ),
    )
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
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.30
    )

    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )

    if cfg.local_rank == 0:
        logger.info('Total Batch Size: {}'.format(cfg.batch_size))

    optimizer_params = {'lr': 5e-5 * 140 / cfg.batch_size, 'betas': (0.9, 0.999), 'eps': 1e-8}

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 62], gamma=0.1)
    trainer = ISTrainerPiClick(model, cfg, model_cfg, loss_cfg,
                               trainset, valset,
                               optimizer='adam',
                               optimizer_params=optimizer_params,
                               layerwise_decay=cfg.layerwise_decay,
                               lr_scheduler=lr_scheduler,
                               checkpoint_interval=[(0, 1), ],
                               image_dump_interval=30,
                               metrics=[AdaptiveIoU()],
                               max_interactive_points=model_cfg.num_max_points,
                               max_num_next_clicks=3,
                               find_unused_parameters=True)
    trainer.run(num_epochs=62, validation=False)
