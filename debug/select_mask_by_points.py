import sys

sys.path.insert(1, '.')

import pickle
from copy import deepcopy
from tools.visual import draw_sample
import numpy as np
import cv2


def get_masks_by_points(sample: dict) -> dict:
    points_pos, points_neg = sample['points'].reshape((2, -1, 3)).astype(int)
    layers = sample['data_info']['mask']
    gt_mask = list()
    for obj_id, info in sample['data_info']['object'].items():
        layer_id, mask_id = info['mapping']
        mask = np.array(layers[:, :, layer_id] == mask_id)
        flag_pos = all((f == -1) or mask[x, y] for x, y, f in points_pos)
        flag_neg = all((f == -1) or (not mask[x, y]) for x, y, f in points_neg)
        if flag_pos and flag_neg:
            gt_mask.append(np.array(deepcopy(mask), dtype=float))
    sample['gt_masks'] = np.array(gt_mask)
    return sample


def main():
    sample = pickle.load(open('debug/data/sample.pkl', 'rb'))
    sample = get_masks_by_points(sample)
    img = draw_sample(sample, mode='gt')
    cv2.imshow('1', img)
    cv2.waitKey()
    pass


if __name__ == '__main__':
    main()
