import pickle
import math
from typing import List, Tuple
import numpy as np
import cv2
from copy import deepcopy

PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
           (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
           (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
           (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
           (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
           (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
           (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
           (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
           (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
           (134, 134, 103), (145, 148, 174), (255, 208, 186),
           (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
           (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
           (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
           (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
           (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
           (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
           (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
           (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
           (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
           (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
           (191, 162, 208)]


def bitmap_to_polygon(bitmap):
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole


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
        # import pdb; pdb.set_trace()
        x, y = (w - text_width) // 2, h + (margin + text_height) // 2
        cv2.putText(img, cap, (x, y), font, font_scale, color, thickness)
        imgs[idx] = img
        # cv2.imwrite(str(idx) + '.jpg', imgs[idx])


def image_concat(imgs: List[np.ndarray],
                 img_per_row: int = 3
                 ):
    r = math.ceil(len(imgs) / img_per_row)
    img_ones = np.ones_like(imgs[0]) * 255

    img_res = cv2.vconcat([
        cv2.hconcat([imgs[i * img_per_row + j] if i * img_per_row + j < len(imgs) else img_ones
                     for j in range(img_per_row)]) for i in range(r)
    ])

    return img_res


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def draw_masks(img, masks, color=None, alpha=0.8):
    """Draw masks on the image

    Args:
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape of (n, 3).
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    for i, mask in enumerate(masks):

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha

    return img


def masks_unsqueeze(mask: np.ndarray) -> np.ndarray:
    vals: list = sorted(np.unique(mask).tolist())
    vals.remove(0)
    return np.array([mask == v for v in vals])


def masks_select(mask: np.ndarray, selected: List[int]) -> np.ndarray:
    vals: list = sorted(np.unique(mask).tolist())
    vals.remove(0)
    selected = list(set.intersection(set(vals), set(selected)))
    return np.array([mask == v for v in selected])


def draw_sample(sample: dict, out_path: str = None, mode: str = 'ori') -> np.ndarray:
    img = np.array(sample['images'].permute((1, 2, 0)).cpu().numpy() * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    if mode == 'ori':
        mask = np.array(np.array(sample['instances'], dtype=int) == 1)
    elif mode == 'gt':
        mask = np.array(np.array(sample['gt_masks'], dtype=int) == 1)
    else:
        raise

    img = draw_masks(img, mask, np.array(list(reversed(PALETTE)), dtype=np.uint8), alpha=0.7)
    points_pos, points_neg = sample['points'].reshape((2, -1, 3)).astype(int)
    for y, x, tag in points_pos:  # red
        if tag == -1: continue
        img = cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
    for y, x, tag in points_neg:  # blue
        if tag == -1: continue
        img = cv2.circle(img, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
    h, w = sample['data_info']['ori_shape']
    img = cv2.resize(img, (w, h))
    if out_path is not None:
        cv2.imwrite(out_path, img)
    return img
