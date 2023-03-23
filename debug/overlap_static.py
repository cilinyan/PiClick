import pickle
from collections import defaultdict
from tqdm import tqdm
from loguru import logger


def static(pkl_path: str = 'data/hannotation.pickle') -> dict:
    anno = pickle.load(open(pkl_path, 'rb'))
    data = defaultdict(int)
    for img, info in tqdm(anno.items()):
        for obj_id, obj_info in info['hierarchy'].items():
            if obj_info is None:
                data['1'] += 1
            elif len(obj_info['children']) == 0:
                data[str(obj_info['node_level'] + 1)] += 1
    return data


def main():
    train_info = static('/data/clyan/ClickClick/datasets/LVIS/train/hannotation.pickle')
    logger.info(f'TRAIN: {train_info}')
    val_info = static('/data/clyan/ClickClick/datasets/LVIS/val/hannotation.pickle')
    logger.info(f'VAL: {val_info}')
    pass


if __name__ == '__main__':
    main()
