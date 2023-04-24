import os
from time import time
import pdb
import numpy as np
import torch
import pickle

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

from loguru import logger


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:
            _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index, **kwargs)
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def select_max_iou_mask(gt_mask, pred_probs, pred_thr):
    if pred_probs.shape[0] == 1:
        return None, 0, None, pred_probs[0]

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


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, ):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image, gt_mask=gt_mask)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs_all = predictor.get_prediction(clicker, )
            max_iou_val, idx_max_iou, mask_max_iou, prob_max_iou = \
                select_max_iou_mask(gt_mask, pred_probs_all, pred_thr)

            pred_probs = prob_max_iou
            predictor.prev_prediction = torch.tensor(prob_max_iou[None, None, :, :],
                                                     dtype=predictor.prev_prediction.dtype,
                                                     device=predictor.prev_prediction.device)

            predictor.set_prev_for_zoom_in(idx_max_iou)

            pred_mask = pred_probs > pred_thr
            iou = utils.get_iou(gt_mask, pred_mask)

            if (callback is not None) and (pred_probs_all.shape[0] != 1):
                callback(image, gt_mask, pred_probs_all, sample_id, click_indx, clicker.clicks_list)

            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
