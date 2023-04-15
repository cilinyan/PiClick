"""
conda activate click && python scripts/evaluate_model_max_iou.py NoBRS --gpu=4 \
  --checkpoint=./weights/models/iter_mask/multimask_despair_base448_cocolvis_itermask/003/checkpoints/056.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
"""

import os
from multiprocessing import Pool
from loguru import logger
from tqdm import tqdm

_CPT_TEMPLATE = './weights/models/iter_mask/multimask_despair_large448_cocolvis_itermask/001/checkpoints/0{}.pth'


def _eval_cpt(cpt_path, gpu_idx):
    os.system('python scripts/evaluate_model_max_iou.py NoBRS --gpu={} '
              '--checkpoint={} '
              '--eval-mode=cvpr '
              '--datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB '
              '--output-tuple'.format(gpu_idx, cpt_path))
    return 1


def error_callback(e):
    logger.error(str(e))


def main():
    cpt_list = list(range(50, 80, 1))
    p = Pool(len(cpt_list))
    pbar = tqdm(total=len(cpt_list))
    for i, cpt_idx in enumerate(cpt_list):
        cpt_path = _CPT_TEMPLATE.format(cpt_idx)
        gpu_idx = i % 8
        p.apply_async(_eval_cpt, args=(cpt_path, gpu_idx), error_callback=error_callback, callback=pbar.update)
    p.close()
    p.join()
    logger.info('Done!')


if __name__ == '__main__':
    main()
