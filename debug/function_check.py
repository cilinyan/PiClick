import cv2
import random
import numpy as np


def get_point_candidates(obj_mask, k=1.7, full_prob=0.0):
    if full_prob > 0 and random.random() < full_prob:
        return obj_mask

    padded_mask = np.pad(obj_mask, ((1, 1), (1, 1)), 'constant')

    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    if k > 0:
        inner_mask = dt > dt.max() / k
        return np.argwhere(inner_mask)
    else:
        prob_map = dt.flatten()
        prob_map /= max(prob_map.sum(), 1e-6)
        click_indx = np.random.choice(len(prob_map), p=prob_map)
        click_coords = np.unravel_index(click_indx, dt.shape)
        return np.array([click_coords])


def point_select_check():
    mask = np.array([[0, 1, 1, 0],
                     [1, 1, 1, 1],
                     [0, 1, 0, 0]])
    print(get_point_candidates(mask, k=1.7))
    pass


if __name__ == '__main__':
    point_select_check()
