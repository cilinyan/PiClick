import os
import math
import cv2
import pickle
import numpy as np
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt
from functools import lru_cache
from collections import defaultdict
from loguru import logger
from copy import deepcopy

"""
https://search-multimodal-1251524319.cos.ap-shanghai.myqcloud.com/clyan/piclick/DAVIS_193_0.jpg
https://search-multimodal-1251524319.cos.ap-shanghai.myqcloud.com/clyan/piclick/DAVIS_166_0.jpg
https://search-multimodal-1251524319.cos.ap-shanghai.myqcloud.com/clyan/piclick/DAVIS_84_0.jpg
https://search-multimodal-1251524319.cos.ap-shanghai.myqcloud.com/clyan/piclick/DAVIS_52_0.jpg
https://search-multimodal-1251524319.cos.ap-shanghai.myqcloud.com/clyan/piclick/PascalVOC_1371_0.jpg
https://search-multimodal-1251524319.cos.ap-shanghai.myqcloud.com/clyan/piclick/PascalVOC_1304_0.jpg
https://search-multimodal-1251524319.cos.ap-shanghai.myqcloud.com/clyan/piclick/PascalVOC_1139_0.jpg
https://search-multimodal-1251524319.cos.ap-shanghai.myqcloud.com/clyan/piclick/PascalVOC_348_0.jpg
"""

_IMAGE_PAIRS = [
    dict(
        simple_click='debug/vis/vis_in/52_0.pkl',
        piclick='debug/vis/vis_in/52_0_n3.pkl',
        indices=((0, 0.49), (2, 0.485))
    ),
    dict(
        simple_click='debug/vis/vis_in/84_0.pkl',
        piclick='debug/vis/vis_in/84_0_n5.pkl',
        indices=((2, 0.30), (4, 0.34))
    ),
    dict(
        simple_click='debug/vis/vis_in/166_0.pkl',
        piclick='debug/vis/vis_in/166_0_n4.pkl',
        indices=((0, 0.45), (2, 0.45), (1, 0.38))
    ),
    dict(
        simple_click='debug/vis/vis_in/193_0.pkl',
        piclick='debug/vis/vis_in/193_0_n3.pkl',
        indices=((0, 0.49), (1, 0.49), (2, 0.28))
    ),
]


def get_iou(gt_mask, pred_mask, target_label=255, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1
    pred_mask = pred_mask == target_label

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


@lru_cache(maxsize=16)
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


def resize(image: np.ndarray, interpolation, width: int = 321):
    h, w = image.shape[:2]
    height = round(width * h / w)
    image = cv2.resize(image, (width, height), interpolation)
    return image


def gt_reset(mask_gt: np.ndarray):
    mask_gt = deepcopy(mask_gt)
    ignore = mask_gt == -1
    mask_gt[ignore] = 255
    return mask_gt.astype(np.uint8)


def main():
    row = len(_IMAGE_PAIRS)
    col = max(len(c['indices']) for c in _IMAGE_PAIRS) + 3
    width = 321
    legends = ['(a) Input', '(b) Ground truth', '(c) Simple Click',
               '(d) Segmentation 1', '(e) Segmentation 2', '(f) Segmentation 3', '(4) Segmentation 3']

    images = list()

    for i, case in enumerate(_IMAGE_PAIRS):
        logger.info('read data from {}'.format(case['piclick']))
        with open(case['simple_click'], 'rb') as file:
            data_sc = pickle.load(file)
        with open(case['piclick'], 'rb') as file:
            data_pc = pickle.load(file)
        # Original Image
        image = data_sc['image']
        x, y = data_sc['clicks_list'][0].coords
        image = cv2.circle(image, (y, x), 6, (0, 255, 0), -1)
        image = resize(image, cv2.INTER_LINEAR, width)
        height = image.shape[0]
        images.append(image)
        # GT Mask
        mask_gt_for_iou = resize(gt_reset(data_sc['gt_mask']), cv2.INTER_NEAREST, width)
        mask_gt = (np.array(data_sc['gt_mask'] == 1, dtype=float) * 255).astype(np.uint8)
        mask_gt = resize(mask_gt, cv2.INTER_NEAREST, width)
        images.append(mask_gt)
        # Simple Click
        mask_sc = (np.array(data_sc['pred_probs'] > 0.49, dtype=float) * 255).astype(np.uint8)
        mask_sc = resize(mask_sc, cv2.INTER_NEAREST, width)
        images.append(mask_sc)
        iou_sc = get_iou(mask_gt_for_iou, mask_sc)
        logger.info('iou of Simple Click: {:.3f}'.format(iou_sc))
        # Piclick
        for idx_mask, thr in case['indices']:
            mask_pci = (np.array(data_pc['pred_probs'][idx_mask] > thr, dtype=float) * 255).astype(np.uint8)
            mask_pci = resize(mask_pci, cv2.INTER_NEAREST, width)
            images.append(mask_pci)
            iou_pci = get_iou(mask_gt_for_iou, mask_pci)
            logger.info('iou of Piclick {}: {:.3f}'.format(idx_mask, iou_pci))
        for _ in range(len(case['indices']) + 3, col):
            mask_zero = np.zeros((height, width), dtype=np.uint8)
            images.append(mask_zero)

    for idx, img in enumerate(images):
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        images[idx] = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    images = [cv2.vconcat(images[i::col]) for i in range(col)]

    plt.figure(figsize=(20, 20))
    plt.rc('font', family='Times New Roman')
    for idx, img in enumerate(images):
        plt.subplot(1, col, idx + 1, fc='b')
        plt.imshow(img)
        plt.axis('off')
        plt.title(legends[idx], y=-.05, fontdict={'size': 20})
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.002, hspace=None)
    # plt.show()
    plt.savefig('debug/vis/cpr_3.pdf')
    pass


if __name__ == '__main__':
    main()
