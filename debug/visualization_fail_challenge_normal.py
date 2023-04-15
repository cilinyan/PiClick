import numpy as np
import os
import math
import cv2
import pickle
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt
from functools import lru_cache
from collections import defaultdict
from loguru import logger
from copy import deepcopy
from operator import add
from functools import reduce

_WORST_CASE = dict(
    data_path='debug/vis/vis0412/_fail_10.pkl'
)
_CHALLENGE_CASE = dict(
    data_path='debug/vis/vis0412/_challenge_7.pkl'
)
_NORMAL_CASE = dict(
    data_path=[
        'debug/vis/vis0412/_normal_5.pkl',
        'debug/vis/vis0412/_normal_71.pkl',
        'debug/vis/vis0412/_normal_18.pkl',
        'debug/vis/vis0412/_normal_25.pkl',
        'debug/vis/vis0412/_normal_21.pkl',
    ]
)


def main():
    # draw_5x2(_CHALLENGE_CASE['data_path'])
    img_list = [
        draw_5x2(_WORST_CASE['data_path']),
        draw_5x2(_CHALLENGE_CASE['data_path']),
        draw_5x3(_NORMAL_CASE['data_path']),
    ]
    h_max = max(img.shape[0] for img in img_list)
    for idx, img in enumerate(img_list):
        w = img.shape[1] * h_max // img.shape[0]
        img_list[idx] = cv2.resize(img, (w, h_max))
        # cv2.copyMakeBorder(img_list[idx], 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img_list[idx] = cv2.cvtColor(img_list[idx], cv2.COLOR_BGR2RGB)
    save_5x3_1x1(img_list, 'debug/vis/vis0412_individual')
    img_list = reduce(add, split_5x3(img_list))

    legends = ['(a) the worst case', '(b) the challenge case', '(c) five normal cases']
    fig, axs = plt.subplots(5, 3,
                            figsize=(30, 20),
                            gridspec_kw={
                                'width_ratios': [img.shape[1] for img in img_list[:3]],
                                'height_ratios': [1] * 5,
                            }, )
    plt.rc('font', family='Times New Roman')
    for idx, (ax, img) in enumerate(zip(axs.flatten(), img_list)):
        ax.imshow(img, interpolation='nearest')
        ax.axis('off')
        if idx >= (5 - 1) * 3:
            ax.set_title(legends[idx % 3], y=-.21, fontdict={'size': 25})
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=-0.75)
    plt.savefig('debug/vis/all_3.pdf')
    pass


def save_5x3_1x1(img_list: list, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    h = img_list[0].shape[0] // 5
    for idx_col, img_col in enumerate(img_list):
        # img_col: 5x2, 5x2, 5x3
        img_split = [img_col[i * h: (i + 1) * h, :, :] for i in range(5)]
        for idx_row, img_row in enumerate(img_split):
            # img_row: 1x2, 1x2, 1x3
            num_sub_col = 3 if idx_col == 2 else 2
            wi = img_row.shape[1] // num_sub_col
            img_mini = [img_row[:, i * wi: (i + 1) * wi, :] for i in range(num_sub_col)]
            for idx_mini, img in enumerate(img_mini):
                out_file = os.path.join(out_dir, f'{idx_row}_{idx_col}_{idx_mini}.png')
                cv2.imwrite(out_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            pass
        pass


def split_5x3(img_list: list):
    res = list()
    h = img_list[0].shape[0] // 5
    for img in img_list:
        img_split = [img[i * h: (i + 1) * h, :, :] for i in range(5)]
        res.append(img_split)
    res = list(zip(*res))
    return res


def draw_5x3(data_list: list):
    image_list = [draw_1x3(data_path) for data_path in data_list]
    image_resize(image_list)
    return cv2.vconcat(image_list)


def draw_1x3(file_path: str, prob_thresh=0.49):
    logger.info(f'file_path: {file_path}')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    image = cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR)
    gt_mask_2 = (np.array(data['gt_mask'] == 1, dtype=float) * 255)[:, :, None].astype(np.uint8).repeat(3, axis=2)
    gt_mask = data['gt_mask']
    clicks_list = data['clicks_list']
    click_indx = data['click_indx'] + 1
    iou, _, _, pred_prob = select_max_iou_mask(gt_mask, data['pred_probs_all'])
    prob_map = draw_probmap(pred_prob)
    image_with_mask = draw_with_blend_and_clicks(image, pred_prob > prob_thresh, clicks_list=clicks_list)
    image = cv2.hconcat([image_with_mask, prob_map, gt_mask_2])
    logger.info(f'    iou: {iou:.3f}, click_indx: {click_indx}')
    return image


def draw_5x2(file_path, prob_thresh=0.49):
    logger.info(f'file_path: {file_path}')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    image = cv2.cvtColor(data[0]['image'], cv2.COLOR_RGB2BGR)
    gt_mask = (np.array(data[0]['gt_mask'] == 1, dtype=float) * 255)[:, :, None].astype(np.uint8).repeat(3, axis=2)
    image_list = [cv2.hconcat([image, gt_mask]), ]
    for i in [0, 1, 2, 3]:
        image = cv2.cvtColor(data[i]['image'], cv2.COLOR_RGB2BGR)
        gt_mask = data[i]['gt_mask']
        click_indx = data[i]['click_indx'] + 1
        clicks_list = data[i]['clicks_list'][:click_indx]
        iou, _, _, pred_prob = select_max_iou_mask(gt_mask, data[i]['pred_probs_all'])
        prob_map = draw_probmap(pred_prob)
        image_with_mask = draw_with_blend_and_clicks(image, pred_prob > prob_thresh, clicks_list=clicks_list)
        image_list.append(cv2.hconcat([image_with_mask, prob_map]))
        logger.info(f'    iou: {iou:.3f}, click_indx: {click_indx}')
    image = cv2.vconcat(image_list)
    return image


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def select_max_iou_mask(gt_mask, pred_probs, pred_thr=0.49):
    if pred_probs.shape[0] == 1:
        pred_prob = pred_probs[0]
        pred_mask = pred_prob > pred_thr
        iou = get_iou(gt_mask, pred_mask)
        return iou, 0, pred_mask, pred_prob

    max_iou = -1
    idx_max_iou = -1
    mask_max_iou = None
    prob_max_iou = None
    for idx, pred_prob in enumerate(pred_probs):
        pred_mask = pred_prob > pred_thr
        iou = get_iou(gt_mask, pred_mask)
        if iou > max_iou:
            idx_max_iou = idx
            max_iou = iou
            mask_max_iou = pred_mask
            prob_max_iou = pred_prob
    return max_iou, idx_max_iou, mask_max_iou, prob_max_iou


def draw_probmap(x):
    return cv2.applyColorMap(np.array(x * 255, dtype=np.uint8), cv2.COLORMAP_HOT)


def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None, pos_color=(0, 255, 0),
                               neg_color=(255, 0, 0), radius=4):
    result = img.copy()

    if mask is not None:
        # palette = get_palette(np.max(mask) + 1)
        palette = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.uint8)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
                 (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def image_resize(imgs: List[np.ndarray], caps: List[str] = None) -> None:
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
        # t = (h - img.shape[0]) // 2
        # b = h - t - img.shape[0]
        # l = (w - img.shape[1]) // 2
        # r = w - l - img.shape[1]
        # img = cv2.copyMakeBorder(img, t, b + margin, l, r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # img = cv2.rectangle(img, (1, 1), (w - 1, h - 1), (229, 235, 178), 2)
        # img = cv2.rectangle(img, (1, 1), (w - 1, h + margin - 1), (229, 235, 178), 2)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        size = cv2.getTextSize(cap, font, font_scale, thickness)
        text_width, text_height = size[0][0], size[0][1]
        # import pdb; pdb.set_trace()
        x, y = (w - text_width) // 2, h + (margin + text_height) // 2
        cv2.putText(img, cap, (x, y), font, font_scale, color, thickness)
        imgs[idx] = img
        # cv2.imwrite(str(idx) + '.jpg', imgs[idx])


if __name__ == '__main__':
    main()
