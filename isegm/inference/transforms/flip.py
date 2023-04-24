import pdb

import torch

from typing import List
from isegm.inference.clicker import Click
from .base import BaseTransform


class AddHorizontalFlip(BaseTransform):
    def transform(self, image_nd, clicks_lists: List[List[Click]]):
        assert len(image_nd.shape) == 4
        # import pdb; pdb.set_trace()
        image_nd = torch.cat([image_nd, torch.flip(image_nd, dims=[3])], dim=0)  # 1x3/4x224x224

        image_width = image_nd.shape[3]
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = [click.copy(coords=(click.coords[0], image_width - click.coords[1] - 1))
                                   for click in clicks_list]
            clicks_lists_flipped.append(clicks_list_flipped)
        clicks_lists = clicks_lists + clicks_lists_flipped

        return image_nd, clicks_lists

    def inv_transform(self, prob_map, **kwargs):
        if kwargs.get('mode', None) == 'multi_mask':  # 2, 7, 224, 224
            # pdb.set_trace()
            assert len(prob_map.shape) == 4 and prob_map.shape[0] == 2  # and prob_map.shape[1] == 7
            prob_map, prob_map_flipped = prob_map[0], prob_map[1]  # 7, 224, 224
            return torch.stack((prob_map, torch.flip(prob_map_flipped, dims=[2])))

        assert len(prob_map.shape) == 4 and prob_map.shape[0] % 2 == 0
        num_maps = prob_map.shape[0] // 2
        prob_map, prob_map_flipped = prob_map[:num_maps], prob_map[num_maps:]

        # pdb.set_trace()

        return 0.5 * (prob_map + torch.flip(prob_map_flipped, dims=[3]))

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def reset(self):
        pass
