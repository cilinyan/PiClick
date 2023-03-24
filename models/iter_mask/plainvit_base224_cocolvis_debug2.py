import sys

import torch

sys.path.insert(1, '.')

from isegm.utils.exp_imports.default import *
from isegm.model.losses import DETRLikeLoss
from tools.visual import draw_sample
from torch.utils.data import DataLoader
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from isegm.model.is_maskformer_model import MaskFormerModel

from collections import defaultdict

MODEL_NAME = 'cocolvis_plainvit_base224'

_PARAMS = dict(
    num_queries=7,
    num_classes=1,
)

device = torch.device('cuda')


def collate_fn(values):
    res = defaultdict(list)
    for value in values:
        for k, v in value.items():
            res[k].append(v)
    res = {
        'images': torch.stack(res['images']),
        'points': torch.stack(res['points']),
        'instances': torch.stack(res['instances']),
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
    import pdb; pdb.set_trace()
    for batch_data in train_data:
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
        orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()
        prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
        net_input = torch.cat((image, prev_output), dim=1)
        output = model(net_input, points)


if __name__ == '__main__':
    main()
