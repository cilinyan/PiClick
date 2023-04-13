from pathlib import Path
import pickle
import random
import numpy as np
import json
import cv2
import torch
from collections import defaultdict
from copy import deepcopy
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class CocoLvisDataset(ISDataset):
    """

    anno_file: dict,
        eg. anno['000000581913'] = {
            "num_instance_masks": 5,
            "hierarchy": {0: None, 1: None, 2: None, 3: None, 4: None}
        }
            anno['000000581921'] = {
            "num_instance_masks": 5,
            "hierarchy" {0: {'children': [1, 4], 'parent': None, 'node_level': 0}, 1: {'children': [], 'parent': 0, 'node_level': 1}, 2: None, 3: None, 4: {'children': [], 'parent': 0, 'node_level': 1}}
        }

    label_data: tuple, (encoded_layers, objs_mapping)
        where, objs_mapping: List[Tuple[int, int]], [(layer_indx, mask_id), ...]
        eg. ([array(..., dtype=uint8), array(..., dtype=uint8), array(..., dtype=uint8)],
             [(0, 4), (1, 2), (2, 1), (1, 1), (1, 3), (0, 1), (0, 3), (0, 2), (0, 5)])
    """

    def __init__(self, dataset_path, split='train', stuff_prob=0.0,
                 allow_list_name=None, anno_file='hannotation.pickle', **kwargs):
        super(CocoLvisDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.split = split
        self._images_path = self._split_path / 'images'
        self._masks_path = self._split_path / 'masks'
        self.stuff_prob = stuff_prob

        with open(self._split_path / anno_file, 'rb') as f:
            self.dataset_samples = sorted(pickle.load(f).items())

        if allow_list_name is not None:
            allow_list_path = self._split_path / allow_list_name
            with open(allow_list_path, 'r') as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self.dataset_samples = [sample for sample in self.dataset_samples
                                    if sample[0] in allow_images_ids]

    def get_sample(self, index) -> DSample:
        image_id, sample = self.dataset_samples[index]
        image_path = self._images_path / f'{image_id}.jpg'

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        packed_masks_path = self._masks_path / f'{image_id}.pickle'
        with open(packed_masks_path, 'rb') as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)

        instances_info = deepcopy(sample['hierarchy'])
        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {'children': [], 'parent': None, 'node_level': 0}
                instances_info[inst_id] = inst_info
            inst_info['mapping'] = objs_mapping[inst_id]

        # if self.stuff_prob > 0 and random.random() < self.stuff_prob:
        #     for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
        #         instances_info[inst_id] = {
        #             'mapping': objs_mapping[inst_id],
        #             'parent': None,
        #             'children': []
        #         }
        # else:
        #     for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
        #         layer_indx, mask_id = objs_mapping[inst_id]
        #         layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
            instances_info[inst_id] = {
                'mapping': objs_mapping[inst_id],
                'parent': None,
                'children': []
            }

        select_range = len(objs_mapping) \
            if self.stuff_prob > 0 and random.random() < self.stuff_prob else sample['num_instance_masks']

        return DSample(image, layers, objects=instances_info, image_id=image_id, select_range=select_range)

    def collate_fn(self, values):
        res = defaultdict(list)
        for value in values:
            for k, v in value.items():
                res[k].append(v)
        res = {
            'images': torch.stack(res['images']),
            'points': torch.tensor(np.array(res['points'])),
            'instances': torch.tensor(np.array(res['instances'])),
            'data_info': res['data_info'],
        }
        return res
