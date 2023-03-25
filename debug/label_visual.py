import os
import os.path as osp
import pickle
import math
from typing import List, Tuple
import numpy as np
import cv2
from copy import deepcopy
import argparse

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


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    image_id = args.id
    img_path = osp.join(args.root, 'images', f'{image_id}.jpg')
    pkl_path = osp.join(args.root, 'masks', f'{image_id}.pickle')
    ann_path = osp.join(args.root, 'hannotation.pickle')

    img = cv2.imread(img_path)
    with open(ann_path, 'rb') as f:
        dataset_samples = pickle.load(f)
    sample = dataset_samples[image_id]

    packed_masks_path = pkl_path
    with open(packed_masks_path, 'rb') as f:
        encoded_layers, objs_mapping = pickle.load(f)
    layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]

    instances_info = deepcopy(sample['hierarchy'])
    for inst_id, inst_info in list(instances_info.items()):
        if inst_info is None:
            inst_info = {'children': [], 'parent': None, 'node_level': 0}
            instances_info[inst_id] = inst_info
        inst_info['mapping'] = objs_mapping[inst_id]

    for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
        instances_info[inst_id] = {
            'mapping': objs_mapping[inst_id],
            'parent': None,
            'children': []
        }

    imgs = [deepcopy(img)] + \
           [draw_masks(deepcopy(img), masks_unsqueeze(l), np.array(PALETTE, dtype=np.uint8), alpha=0.8) for l in layers]

    image_padding(imgs, ['original'] + [str(i) for i, _ in enumerate(layers)])

    img_show = image_concat(imgs, 3)

    cv2.imwrite(osp.join(args.out, 'all.jpg'), img_show)

    img_ins = list()
    cap_ins = list()
    for inst_id, info in instances_info.items():
        layer_id, mask_id = info['mapping']
        img_ins.append(
            draw_masks(deepcopy(img), masks_select(layers[layer_id], [mask_id]),
                       np.array(PALETTE, dtype=np.uint8), alpha=1.0)
        )
        cap_ins.append(f'i{inst_id}_l{layer_id}_m{mask_id}')
    image_padding(img_ins, cap_ins)
    img_ins = image_concat(img_ins, 3)

    cv2.imwrite(osp.join(args.out, 'split.jpg'), img_ins)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=str)
    parser.add_argument('--root', type=str, default='datasets/LVIS/train')
    parser.add_argument('--out', type=str, default='debug/vis/')
    return parser.parse_args()


if __name__ == '__main__':
    main()
