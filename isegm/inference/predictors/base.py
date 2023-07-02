import pdb
import torch
from copy import deepcopy
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from isegm.inference.transforms import ZoomIn
from isegm.inference import utils
from isegm.model.modeling.maskformer_helper.mask_hungarian_assigner import MaskHungarianAssigner
from isegm.model.modeling.maskformer_helper.mask_pseudo_sampler import MaskPseudoSampler
import numpy as np
from typing import List
import math
import cv2
import time
import os
import os.path as osp
from loguru import logger

_MATCH_CFG: dict = dict(
    assigner=dict(type='MaskHungarianAssigner',
                  cls_cost=dict(type='DistCost', weight=1.0),
                  mask_cost=dict(type='FocalLossCost', weight=20.0, binary_input=True),
                  dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
    sampler=dict(type='MaskPseudoSampler')
)


def image_padding(imgs: List[np.ndarray], caps: List[str] = None) -> None:
    if caps is None: caps = [str(i) for i in range(len(imgs))]
    shapes = [img.shape[:2] for img in imgs]
    h = max(shape[0] for shape in shapes)
    w = max(shape[1] for shape in shapes)

    h, w = h + 3, max(w + 3, 240)

    font = cv2.FONT_HERSHEY_DUPLEX
    margin = 40
    font_scale = 1
    thickness = 2
    color = (178, 178, 178)

    for idx, (img, cap) in enumerate(zip(imgs, caps)):  # top, bottom, left, right
        t = (h - img.shape[0]) // 2
        b = h - t - img.shape[0]
        l = (w - img.shape[1]) // 2
        r = w - l - img.shape[1]
        img = cv2.copyMakeBorder(img, t, b + margin, l, r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img = cv2.rectangle(img, (1, 1), (w - 1, h - 1), (229, 235, 178), 2)
        # img = cv2.rectangle(img, (1, 1), (w - 1, h + margin - 1), (229, 235, 178), 2)
        size = cv2.getTextSize(cap, font, font_scale, thickness)
        text_width, text_height = size[0][0], size[0][1]
        x, y = (w - text_width) // 2, h + (margin + text_height) // 2
        cv2.putText(img, cap, (x, y), font, font_scale, color, thickness)
        imgs[idx] = img


def image_concat(imgs: List[np.ndarray],
                 img_per_row: int = 3
                 ):
    img_per_row = min(img_per_row, len(imgs))
    r = math.ceil(len(imgs) / img_per_row)
    img_ones = np.ones_like(imgs[0]) * 255

    img_res = cv2.vconcat([
        cv2.hconcat([imgs[i * img_per_row + j] if i * img_per_row + j < len(imgs) else img_ones
                     for j in range(img_per_row)]) for i in range(r)
    ])

    return img_res


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 **kwargs):

        self.gt_mask_debug = None
        self.image_debug = None
        self.mask_dir = osp.join('./debug/tmpdata', str(time.time()).replace('.', '_'))
        os.makedirs(self.mask_dir, exist_ok=True)

        self.prev_prediction_for_zoom_in = None
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

        assert _MATCH_CFG['assigner']['type'] == 'MaskHungarianAssigner'
        self.assigner = MaskHungarianAssigner(**_MATCH_CFG['assigner'])
        assert _MATCH_CFG['sampler']['type'] == 'MaskPseudoSampler'
        self.sampler = MaskPseudoSampler()

    def set_input_image(self, image, **kwargs):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        if 'gt_mask' in kwargs:
            self.gt_mask_debug = deepcopy(image)
            self.image_debug = deepcopy(image)
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_prev_for_zoom_in(self, idx):
        pred_logits = self.prev_prediction_for_zoom_in[:, idx:idx + 1, :, :]
        for i, t in enumerate(self.transforms):
            if isinstance(t, ZoomIn):
                t.set_prev_probs(pred_logits)

    def mask_match_visual(self, cls_scores, masks, output_path, prefix: str = ""):
        # split
        score_0, score_1 = cls_scores[0], cls_scores[1]
        mask_0, mask_1 = masks[0].cpu().numpy(), masks[1].cpu().numpy()
        caps_0 = [f'{prefix}_0_{i}_score_{s.item():.2f}' for i, s in enumerate(score_0)]
        caps_1 = [f'{prefix}_1_{i}_score_{s.item():.2f}' for i, s in enumerate(score_1)]
        # cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        imgs_0 = [cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT) for x in mask_0]
        imgs_1 = [cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT) for x in mask_1]
        # concat
        caps = caps_0 + caps_1
        imgs = imgs_0 + imgs_1
        image_padding(imgs, caps)
        img = image_concat(imgs, img_per_row=7)
        cv2.imwrite(output_path, img)
        pass

    def flipped_mask_match(self, cls_scores, masks_pred, rank_scores, onehot_scores,
                           seg_thr=.49, vis_flag: bool = False, max_score: bool = False):
        """
        Args:
            cls_scores:    shape 2x7x2 or 2x7
            masks_pred:    shape 2x7x224x224
            rank_scores:   shape 2x7
            onehot_scores: shape 2x7
        """
        # time
        time_stamp = str(time.time()).replace('.', '_')
        # score softmax
        if len(cls_scores.shape) == 3:
            cls_scores = cls_scores.softmax(dim=-1)[..., 0]  # 2x7x2 -> 2x7
        # mask
        masks_pred = self.transforms[-1].inv_transform(masks_pred, mode='multi_mask')  # Flip
        masks = torch.sigmoid(masks_pred)  # Sigmoid
        masks = (masks > seg_thr).to(masks).float()
        if vis_flag:
            self.mask_match_visual(cls_scores, masks, prefix='ori',
                                   output_path=osp.join(self.mask_dir, f'{time_stamp}_ori.jpg'))
        # split
        score_0, score_1 = cls_scores[0], cls_scores[1]
        rank_score_0, rank_score_1 = rank_scores[0], rank_scores[1]
        onehot_score_0, onehot_score_1 = onehot_scores[0], onehot_scores[1]
        mask_0, mask_1 = masks[0], masks[1].long()
        # assign and sample: 0 -> 1
        assigned_gt_inds = self.assigner.assign_match(score_0, mask_0, score_1, mask_1)
        score_0 = score_0[assigned_gt_inds, ...]
        mask_0 = mask_0[assigned_gt_inds, ...]

        mask_0_p, mask_1_p = masks_pred[0], masks_pred[1]
        mask_0_p = mask_0_p[assigned_gt_inds, ...]
        score_0_p, schore_1_p = score_0, score_1
        score_0_p = score_0_p[assigned_gt_inds, ...]
        rank_score_0_p, rank_score_1_p = rank_score_0, rank_score_1
        rank_score_0_p = rank_score_0_p[assigned_gt_inds, ...]
        onehot_score_0_p, onehot_score_1_p = onehot_score_0, onehot_score_1
        onehot_score_0_p = onehot_score_0_p[assigned_gt_inds, ...]

        if vis_flag:
            self.mask_match_visual(torch.stack([score_0, score_1]), torch.stack([mask_0, mask_1]), prefix='match',
                                   output_path=osp.join(self.mask_dir, f'{time_stamp}_match.jpg'))

        masks_p_r = 0.5 * (mask_0_p[None, ...] + mask_1_p[None, ...])  # [1, 7, 448, 448]
        masks_p = torch.stack([mask_0_p, mask_1_p])
        masks_p = self.transforms[-1].inv_transform(masks_p, mode='multi_mask')  # Flip, [2, 7, 448, 448]

        if max_score:
            final_score_0 = 6 * rank_score_0_p + 1 * onehot_score_0_p
            final_score_1 = 6 * rank_score_1_p + 1 * onehot_score_1_p
            final_score = final_score_0 + final_score_1  # [7, ]
            idx_max = final_score.argmax()
            masks_p_r = masks_p_r[:, idx_max:idx_max + 1, ...]
            masks_p = torch.stack([mask_0_p[idx_max:idx_max + 1], mask_1_p[idx_max:idx_max + 1]])
            masks_p = self.transforms[-1].inv_transform(masks_p, mode='multi_mask')  # Flip
            pass

        return masks_p, masks_p_r

    def get_prediction(self, clicker, prev_mask=None, max_score=False):
        clicks_list = clicker.get_clicks()

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image

        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(input_image, [clicks_list])
        assert self.with_flip
        instances = self._get_prediction(image_nd, clicks_lists, is_image_changed)
        pred_logits, pred_rignt_ = self.flipped_mask_match(*instances, max_score=max_score)

        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True, size=image_nd.size()[2:])

        if pred_rignt_ is None:
            self.prev_prediction_for_zoom_in = deepcopy(prediction)
        else:
            self.prev_prediction_for_zoom_in = deepcopy(pred_rignt_)

        for t in reversed(self.transforms):
            if isinstance(t, ZoomIn):
                prediction = t.inv_transform(prediction, save_delay=True)
            else:
                prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        return prediction.cpu().numpy()[0]

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed, **kwargs):
        points_nd = self.get_points_nd(clicks_lists)
        # import pdb; pdb.set_trace()
        return self.net(image_nd, points_nd, last_layer=True)['instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        total_clicks = torch.tensor(total_clicks, device=self.device)

        return total_clicks

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']


def select_max_iou_mask(gt_mask, pred_probs, pred_thr):
    max_iou = -1
    idx_max_iou = -1
    mask_max_iou = None
    prob_max_iou = None
    for idx, pred_prob in enumerate(pred_probs):
        pred_mask = pred_prob > pred_thr
        iou = utils.get_iou(gt_mask, pred_mask)
        if iou > max_iou:
            idx_max_iou = idx
            max_iou = iou
            mask_max_iou = pred_mask
            prob_max_iou = pred_prob
    return max_iou, idx_max_iou, mask_max_iou, prob_max_iou
