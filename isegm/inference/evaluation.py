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

_VIS_TYPE_MAX = {
    'normal': 128,
    'fail': 16,
    'challenge': 16,
}


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:

            if kwargs['max_clicks'] == 100 and all(v <= 0 for v in _VIS_TYPE_MAX.values()):
                raise ValueError(
                    'only when visualization is enabled, max_clicks can be 100, and when the number of visualization is 0, raise an error to return the result quickly'
                )

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
                    sample_id=None, callback=None, max_iou=False, output_tuple=False):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        data_record = list()
        predictor.set_input_image(image, gt_mask=gt_mask)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            if not max_iou:  # SimpleClick 测试模式
                pred_probs = predictor.get_prediction(clicker, output_tuple=output_tuple)

                pred_mask = pred_probs > pred_thr
                iou = utils.get_iou(gt_mask, pred_mask)

                # pdb.set_trace()
                if (callback is not None) and (iou < 0.8):
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)
            else:  # 输出所有可能的 masks
                pred_probs_all = predictor.get_prediction(clicker, output_tuple=output_tuple)
                max_iou_val, idx_max_iou, mask_max_iou, prob_max_iou = \
                    select_max_iou_mask(gt_mask, pred_probs_all, pred_thr)

                pred_probs = prob_max_iou
                predictor.prev_prediction = torch.tensor(prob_max_iou[None, None, :, :],
                                                         dtype=predictor.prev_prediction.dtype,
                                                         device=predictor.prev_prediction.device)

                predictor.set_prev_for_zoom_in(idx_max_iou)

                pred_mask = pred_probs > pred_thr
                iou = utils.get_iou(gt_mask, pred_mask)

                # pdb.set_trace()
                if (callback is not None) and (pred_probs_all.shape[0] != 1):
                    callback(image, gt_mask, pred_probs_all, sample_id, click_indx, clicker.clicks_list)

                # only when visual the max clicks will be set to 100
                if max_clicks == 100:
                    # VISUAL 100
                    data_record.append(dict(
                        image=image,
                        gt_mask=gt_mask,
                        pred_probs_all=pred_probs_all,
                        sample_id=sample_id,
                        click_indx=click_indx,
                        clicks_list=clicker.clicks_list,
                        iou=iou,
                    ))
                    os.makedirs('/data/clyan/vis', exist_ok=True)
                    if click_indx <= 4 and iou > 0.9 and _VIS_TYPE_MAX['normal'] > 0:
                        normal_idx = _VIS_TYPE_MAX['normal']
                        _VIS_TYPE_MAX['normal'] -= 1
                        with open('/data/clyan/vis/normal_{}.pkl'.format(normal_idx), 'wb') as f:
                            pickle.dump(data_record[-1], f)
                    if click_indx == 99 and iou < 0.8 and _VIS_TYPE_MAX['fail'] > 0:
                        fail_idx = _VIS_TYPE_MAX['fail']
                        _VIS_TYPE_MAX['fail'] -= 1
                        with open('/data/clyan/vis/fail_{}.pkl'.format(fail_idx), 'wb') as f:
                            pickle.dump([data_record[i] for i in [0, 9, 19, 99]], f)
                    if click_indx == 99 and (0.8 < iou < 0.9) and _VIS_TYPE_MAX['challenge'] > 0:
                        challenge_idx = _VIS_TYPE_MAX['challenge']
                        _VIS_TYPE_MAX['challenge'] -= 1
                        with open('/data/clyan/vis/challenge_{}.pkl'.format(challenge_idx), 'wb') as f:
                            pickle.dump([data_record[i] for i in [0, 9, 19, 99]], f)

            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        del data_record

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
