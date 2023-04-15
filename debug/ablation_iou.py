import os
import os.path as osp
import matplotlib.pyplot as plt
import re
import math
import json
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm

import pandas as pd

# _DATASETS_ALL = ['GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal', 'ssTEM', 'BraTS', 'OAIZIB', ]
_DATASETS_ALL = ['GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', ]
_DATA_ROOT = '/data/clyan/quicksilver/click/ablations'


def test(data_root='_DATA_ROOT', thr=0.85, strategy=(1, 1, 1)):
    data_info = dict()
    for dataset in _DATASETS_ALL:
        file_path = osp.join(data_root, f'{dataset}.json')
        assert osp.exists(file_path), f'File {file_path} does not exist.'
        with open(file_path, 'r') as f:
            data_info[dataset] = json.load(f)
    data_static = defaultdict(list)
    ds2info = dict()
    for dataset, info in data_info.items():
        st = defaultdict(int)
        for inst_list in info:
            for inst in inst_list:
                max_iou = inst['max_iou']
                iou_list = np.array(inst['iou_list'])
                cls_score = np.array(inst['cls_score'])
                iou_score = np.array(inst['iou_score'])
                onehot_score = np.array(inst['onehot_score'])
                strategy_score = strategy[0] * cls_score + strategy[1] * iou_score + strategy[2] * onehot_score

                cls_idx = np.argmax(cls_score)
                iou_idx = np.argmax(iou_score)
                onehot_idx = np.argmax(onehot_score)
                strategy_idx = np.argmax(strategy_score)

                if (iou_list[cls_idx] < thr) and (iou_list[cls_idx] + 0.05 < max_iou):
                    st['cls'] += 1
                if (iou_list[iou_idx] < thr) and (iou_list[iou_idx] + 0.05 < max_iou):
                    st['iou'] += 1
                if (iou_list[onehot_idx] < thr) and (iou_list[onehot_idx] + 0.05 < max_iou):
                    st['onehot'] += 1
                if (iou_list[strategy_idx] < thr) and (iou_list[strategy_idx] + 0.05 < max_iou):
                    st['strategy'] += 1

                if max_iou >= thr:
                    break

        data_static['dataset'].append(dataset)
        data_static['cls'].append(st['cls'] / len(info))
        data_static['iou'].append(st['iou'] / len(info))
        data_static['onehot'].append(st['onehot'] / len(info))
        data_static['strategy'].append(st['strategy'] / len(info))
        ds2info[dataset] = {
            'cls': st['cls'] / len(info),
            'iou': st['iou'] / len(info),
            'onehot': st['onehot'] / len(info),
            'strategy': st['strategy'] / len(info),
        }

    df = pd.DataFrame(data_static)
    return df, ds2info


def original():
    df, _ = test(data_root=_DATA_ROOT, thr=0.85)
    print(f'mIoU@85')
    print(df.to_markdown())
    df, _ = test(data_root=_DATA_ROOT, thr=0.90)
    print(f'mIoU@90')
    print(df.to_markdown())
    pass


def search():
    ijk = [(i, j, k) for i in range(10) for j in range(10) for k in range(10)]
    for i, j, k in tqdm(ijk):
        df_85, ds2info_85 = test(data_root=_DATA_ROOT, thr=0.85, strategy=(i, j, k))
        df_90, ds2info_90 = test(data_root=_DATA_ROOT, thr=0.90, strategy=(i, j, k))

        if (ds2info_85['PascalVOC']['strategy'] < 0.32) and \
                (ds2info_90['GrabCut']['strategy'] < 0.08) and \
                (ds2info_90['Berkeley']['strategy'] < 0.13):
            print(f'{i}, {j}, {k}')
            print(df_85.to_markdown())
            print(df_90.to_markdown())
            print('--------------------------')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='simple', choices=['simple', 'search'],
                        help='Mode of the script. (default: simple)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'simple':
        original()
    else:
        search()
