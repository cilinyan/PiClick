import random
import pickle
import numpy as np
import torch
from torchvision import transforms
from .points_sampler import MultiPointSampler
from .sample import DSample
from copy import deepcopy
from loguru import logger
import math
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")


class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        self.to_tensor = transforms.ToTensor()

        self.dataset_samples = None

    def _get_item(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        sample = self.augment_sample(sample)

        sample.remove_small_objects(self.min_object_area)

        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())
        mask = self.points_sampler.selected_mask

        data_info = deepcopy(sample.data_info)
        data_info['image_id'] = sample.image_id
        data_info['select_range'] = sample.select_range
        data_info['sample_object_ids'] = sample.sample_object_ids

        output = {
            'images': self.to_tensor(sample.image),
            'points': points.astype(np.float32),
            'instances': mask,
            'data_info': data_info,
        }

        if self.with_image_info:
            output['image_info'] = sample.sample_id

        return output

    def __getitem__(self, index):
        logger.debug(f'index:           {index}')
        logger.debug(f'self.actual_len: {self.actual_len}')
        index = index if index < self.actual_len else random.randrange(0, self.actual_len)
        try:
            return self._get_item(index)
        except Exception as e:
            logger.debug(f'fail to read {index}, with error: {e}')
            return self.__getitem__(random.randrange(0, self.actual_len))

    def augment_sample(self, sample) -> DSample:
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (self.keep_background_prob < 0.0 or
                           random.random() < self.keep_background_prob)
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    @property
    def actual_len(self):
        return self.get_samples_number()

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        return self.actual_len

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores
