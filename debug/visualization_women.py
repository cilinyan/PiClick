import os
import os.path as osp
import math
import cv2
import pickle
import numpy as np
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt
from functools import lru_cache
from collections import defaultdict

_IMG_INFO = dict(
    simple_click='debug/vis/c/Berkeley/26_0.pkl',
    piclick='debug/vis/c/Berkeley/26_0_n5.pkl',
    indices=((0, 0.49), (2, 0.485))
)
_OUT_DIR = 'debug/vis/women_images/'


def main(case: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(case['simple_click'], 'rb') as file:
        data_sc = pickle.load(file)
    with open(case['piclick'], 'rb') as file:
        data_pc = pickle.load(file)
    # Original Image
    image = cv2.cvtColor(data_sc['image'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(osp.join(out_dir, 'image_original.jpg'), image)
    x, y = data_sc['clicks_list'][0].coords
    image = cv2.circle(image, (y, x), 7, (0, 255, 0), -1)
    cv2.imwrite(osp.join(out_dir, 'image_with_click.jpg'), image)
    click = np.zeros_like(image)
    click = cv2.circle(click, (y, x), 7, (255, 255, 255), -1)
    cv2.imwrite(osp.join(out_dir, 'click_only.jpg'), click)
    # GT Mask
    mask_gt = (np.array(data_sc['gt_mask'] == 1, dtype=float) * 255).astype(np.uint8)
    cv2.imwrite(osp.join(out_dir, 'mask_gt.jpg'), mask_gt)
    # Simple Click
    mask_sc = (np.array(data_sc['pred_probs'] > 0.49, dtype=float) * 255).astype(np.uint8)
    cv2.imwrite(osp.join(out_dir, 'mask_simpleclick.jpg'), mask_sc)
    for idx_mask, thr in case['indices']:
        mask_pci = (np.array(data_pc['pred_probs'][idx_mask] > thr, dtype=float) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(out_dir, f'mask_piclick_{idx_mask}.jpg'), mask_pci)


if __name__ == '__main__':
    main(_IMG_INFO, _OUT_DIR)
