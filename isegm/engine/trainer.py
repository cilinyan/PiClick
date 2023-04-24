import os
import random
import logging

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay

from ..model.modeling.maskformer_helper.mask_hungarian_assigner import MaskHungarianAssigner
from ..model.modeling.maskformer_helper.mask_pseudo_sampler import MaskPseudoSampler
from ..model.modeling.maskformer_helper.misc import multi_apply
from isegm.data.multi_sample import DistributedSampler

from collections import defaultdict
from loguru import logger
import pdb
from copy import deepcopy
from typing import Union
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

TRAIN_CFG: dict = dict(
    assigner=dict(type='MaskHungarianAssigner',
                  cls_cost=dict(type='ClassificationCost', weight=1.0),
                  mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
                  dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
    sampler=dict(type='MaskPseudoSampler')
)


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


def output_batch_no_first(output):
    output['instances'] = output['instances'][0].transpose(0, 1), output['instances'][1].transpose(0, 1)
    return output


def choice_mask(labels_list, mask_preds_list):
    masks_choice = list()
    for labels, mask_preds in zip(labels_list, mask_preds_list):
        labels = labels.cpu().numpy()
        indexes = [i for i, label in enumerate(labels) if label == 0]
        idx = random.choice(indexes)
        masks_choice.append(mask_preds[idx:idx + 1])
    return torch.stack(masks_choice)


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


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 num_queries=7,
                 num_classes=1,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=20,
                 checkpoint_interval=1,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 find_unused_parameters=False
                 ):
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        if self.is_master:
            logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
            logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        # 给每个rank对应的进程分配训练的样本索引
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(valset)

        # 将样本索引每batch_size个元素组成一个list
        self.train_batch_sampler = torch.utils.data.BatchSampler(self.train_sampler, cfg.batch_size, drop_last=True)

        self.distributed_sampler = DistributedSampler(dataset=trainset,
                                                      num_replicas=cfg.world_size,
                                                      rank=cfg.gpu,
                                                      shuffle=True,
                                                      round_up=True,
                                                      seed=0,
                                                      )

        self.train_data = DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            sampler=self.distributed_sampler,
            # drop_last=True,
            pin_memory=True,
            num_workers=cfg.workers,
            collate_fn=trainset.collate_fn,
        )

        self.val_data = DataLoader(
            valset, batch_size=cfg.val_batch_size,
            sampler=self.val_sampler,
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers,
            collate_fn=valset.collate_fn,
        )

        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)
        self.model = self._load_weights(model)

        self.net = torch.nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[cfg.gpu],
                                                             find_unused_parameters=find_unused_parameters,
                                                             )

        if self.is_master:
            logger.info(self.model)
            logger.info(get_config_repr(model._config))
            logger.info(f'batch size each gpu:    {cfg.batch_size}')
            logger.info(f'total data of datasets: {len(trainset)}')

        self.device = cfg.device
        self.net = self.net.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

        self.assigner = MaskHungarianAssigner(**TRAIN_CFG['assigner'])
        self.sampler = MaskPseudoSampler()

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        if self.is_master:
            logger.info(f'Starting Epoch: {start_epoch}')
            logger.info(f'Total Epochs: {num_epochs}')

        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH), flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            # self.train_sampler.set_epoch(epoch)
            self.distributed_sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100) if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = self.batch_forward(batch_data)

            self.optim.zero_grad()
            loss.backward()
            loss = reduce_value(loss, average=True)
            self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(
                    tag=f'{log_prefix}States/learning_rate',
                    value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                    global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss / (i + 1):.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        torch.cuda.synchronize(self.device)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)
            # model.module
            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss / (i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            logger.debug('images:    {}'.format(batch_data['images'].shape))
            logger.debug('instances: {}'.format(batch_data['instances'].shape))
            logger.debug('points:    {}'.format(batch_data['points'].shape))
            logger.debug('data_info: {}'.format(len(batch_data['data_info'])))
            batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']

            orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]

            last_click_indx = None

            with torch.no_grad():
                num_iters = random.randint(0, self.max_num_next_clicks)

                for click_indx in range(num_iters):
                    last_click_indx = click_indx

                    self.net.eval()

                    if self.click_models is None or click_indx >= len(self.click_models):
                        eval_model = self.net
                    else:
                        eval_model = self.click_models[click_indx]

                    net_input = torch.cat((image, prev_output), dim=1) if self.net.module.with_prev_mask else image
                    output = eval_model.module.forward_for_iter(net_input, gt_mask, points, batch_data)
                    prev_output = torch.sigmoid(output)

                    logger.debug('prev_output: {}'.format(prev_output.shape))
                    logger.debug('points:      {}'.format(points.shape))
                    logger.debug('orig_gt_mask:{}'.format(orig_gt_mask.shape))

                    points = get_next_points(prev_output, orig_gt_mask, points, click_indx + 1)


                if self.net.module.with_prev_mask and self.prev_mask_drop_prob > 0 and last_click_indx is not None:
                    zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                    prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

            self.net.train()

            batch_data['points'] = points

            # 根据当前点生成 gt masks
            gt_masks = \
                [torch.tensor(get_masks_by_points(p, i, g)).long().to(self.device) for p, i, g in
                 zip(batch_data['points'], batch_data['data_info'], gt_mask)]

            net_input = torch.cat((image, prev_output), dim=1) if self.net.module.with_prev_mask else image
            output = self.net(net_input, points, batch_first=True, train_mode=True)
            output = output_batch_no_first(output)

            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (output['instances'], gt_masks))
            loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                                 lambda: (output['instances_aux'], batch_data['instances']))

            output['instances'] = output['single_masks']

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs), *(batch_data[x] for x in m.gt_outputs))
        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        points = points.detach().cpu().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (0, 0, 255))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask, predicted_mask)).astype(np.uint8)

        _save_image('instance_segmentation', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            try:
                checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
                assert len(checkpoints) == 1
                checkpoint_path = checkpoints[0]
            except Exception as e:
                checkpoint_path = 'weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid/002/checkpoints/040.pth'
                logger.error(f'Use {checkpoint_path}')
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0


def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)


def select_max_score_mask(cls_scores_list, mask_preds_list):
    cls_scores_list = cls_scores_list[-1]
    mask_preds_list = mask_preds_list[-1]
    indexes = torch.argmax(cls_scores_list.softmax(dim=-1)[:, :, 0], dim=-1)
    max_scores_masks = list()
    for i, m in zip(indexes, mask_preds_list):
        max_scores_masks.append(m[i:i + 1])
    max_scores_masks = torch.stack(max_scores_masks)
    return max_scores_masks
