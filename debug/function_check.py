import cv2
import random
import torch
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


def select_max_score_mask_ori(cls_scores_list, mask_preds_list, batch_first: bool = False):
    if batch_first:
        cls_scores_list = cls_scores_list.transpose(0, 1)
        mask_preds_list = mask_preds_list.transpose(0, 1)
    cls_scores_list = cls_scores_list[-1]
    mask_preds_list = mask_preds_list[-1]
    indexes = torch.argmax(cls_scores_list.softmax(dim=-1)[:, :, 0], dim=-1)
    max_scores_masks = list()
    for i, m in zip(indexes, mask_preds_list):
        max_scores_masks.append(m[i:i + 1])
    max_scores_masks = torch.stack(max_scores_masks)
    return max_scores_masks


def select_max_score_mask(cls_scores_list, mask_preds_list, batch_first: bool = False):
    if batch_first:
        cls_scores_list = cls_scores_list[:, -1, ...]
        mask_preds_list = mask_preds_list[:, -1, ...]
    else:
        cls_scores_list = cls_scores_list[-1]
        mask_preds_list = mask_preds_list[-1]
    _, _, h, w = mask_preds_list.shape
    indexes = torch.argmax(cls_scores_list.softmax(dim=-1)[:, :, 0], dim=-1)
    indexes = indexes.reshape(-1, 1, 1, 1).repeat(1, 1, *mask_preds_list.shape[-2:])
    max_scores_masks = torch.gather(mask_preds_list, 1, indexes)
    return max_scores_masks


def max_to_one(x: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `x` of shape [batch_size, num_preds], for each row xi of x,
    set the maximum value of xi to 1, and the rest to 0.
    Returns the resulting tensor with the same shape as x.
    """
    # Find the maximum value along the last dimension (num_preds)
    max_values, _ = torch.max(x, dim=-1, keepdim=True)

    # Create a mask of the same shape as x, where 1 corresponds to the maximum value
    mask = torch.eq(x, max_values).type(torch.FloatTensor)

    return mask


def max_min_replace(x, a, b):
    max_val, _ = torch.max(x, dim=1, keepdim=True)
    mask = torch.eq(x, max_val)
    out = torch.where(mask, torch.ones_like(x) * a, torch.ones_like(x) * b)
    return out


def main():
    # cls_scores_list = torch.randn((32, 6, 7, 2))
    # mask_preds_list = torch.randn((32, 6, 7, 224, 224))
    # m1 = select_max_score_mask_ori(cls_scores_list, mask_preds_list)
    # m2 = select_max_score_mask(cls_scores_list, mask_preds_list)
    # print(torch.equal(m1, m2))
    x = torch.tensor([[0.1, 0.5, 0.3], [0.2, 0.2, 0.6]])
    y = max_min_replace(x, 0.9, 0.1)
    print(y)
    pass


if __name__ == '__main__':
    main()
