import os
import os.path as osp
import matplotlib.pyplot as plt
import re
import math
import json
import numpy as np
from collections import defaultdict

import pandas as pd

_DATASETS_ALL = ['GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal', 'ssTEM', 'BraTS', 'OAIZIB', ]


def test_argmax(data_root='ablation_rank'):
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
            iou_list = np.array(inst['iou_list'])
            cls_score = np.array(inst['cls_score'])
            iou_score = np.array(inst['iou_score'])
            onehot_score = np.array(inst['onehot_score'])

            gt_idx = np.argmax(iou_list)
            cls_idx = np.argmax(cls_score)
            iou_idx = np.argmax(iou_score)
            onehot_idx = np.argmax(onehot_score)

            st['cls'] += int(cls_idx == gt_idx)
            st['iou'] += int(iou_idx == gt_idx)
            st['onehot'] += int(onehot_idx == gt_idx)

        data_static['dataset'].append(dataset)
        data_static['cls'].append(st['cls'] / len(info))
        data_static['iou'].append(st['iou'] / len(info))
        data_static['onehot'].append(st['onehot'] / len(info))

    df = pd.DataFrame(data_static)
    print(df.to_markdown())
    pass


if __name__ == '__main__':
    test_argmax()
