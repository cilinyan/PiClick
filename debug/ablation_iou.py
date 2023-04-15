import os
import os.path as osp
import matplotlib.pyplot as plt
import re
import math
import json
import numpy as np
from collections import defaultdict

import pandas as pd

# _DATASETS_ALL = ['GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal', 'ssTEM', 'BraTS', 'OAIZIB', ]
_DATASETS_ALL = ['GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', ]
_DATA_ROOT = '/data/clyan/quicksilver/click/ablations'


def test(data_root='_DATA_ROOT', thr=0.85):
    data_info = dict()
    for dataset in _DATASETS_ALL:
        file_path = osp.join(data_root, f'{dataset}.json')
        assert osp.exists(file_path), f'File {file_path} does not exist.'
        with open(file_path, 'r') as f:
            data_info[dataset] = json.load(f)
    data_static = defaultdict(list)
    for dataset, info in data_info.items():
        st = defaultdict(int)
        for inst in info:
            max_iou = inst['max_iou']
            iou_list = np.array(inst['iou_list'])
            cls_score = np.array(inst['cls_score'])
            iou_score = np.array(inst['iou_score'])
            onehot_score = np.array(inst['onehot_score'])

            cls_idx = np.argmax(cls_score)
            iou_idx = np.argmax(iou_score)
            onehot_idx = np.argmax(onehot_score)

            if (iou_list[cls_idx] < thr) and (iou_list[cls_idx] - 0.05 < max_iou):
                st['cls'] += 1
            if (iou_list[iou_idx] < thr) and (iou_list[iou_idx] - 0.05 < max_iou):
                st['iou'] += 1
            if (iou_list[onehot_idx] < thr) and (iou_list[onehot_idx] - 0.05 < max_iou):
                st['onehot'] += 1

        data_static['dataset'].append(dataset)
        data_static['cls'].append(st['cls'] / len(info))
        data_static['iou'].append(st['iou'] / len(info))
        data_static['onehot'].append(st['onehot'] / len(info))

    df = pd.DataFrame(data_static)
    print(f'IoU@{thr}')
    print(df.to_markdown())
    pass


if __name__ == '__main__':
    test(data_root=_DATA_ROOT, thr=0.85)
    test(data_root=_DATA_ROOT, thr=0.9)
