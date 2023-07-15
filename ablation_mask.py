import os
import argparse

r"""
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59566 --use_env train.py \
  models/iter_mask/piclick_base448_cocolvis_itermask_5m.py \
  --batch-size=136 \
  --ngpus=8 
"""


# def get_args():
#     parser = argparse.ArgumentParser(description='Train a detector')
#     parser.add_argument('--ngpu', default=8)
#     return parser.parse_args()


def main():
    # args = get_args()
    # ngpu = args.ngpu
    # batch_size = 17 * ngpu
    for i in [5, 3, 1, 6, 4, 2, ]:
        cmd = f"python -m torch.distributed.launch --nproc_per_node=8 --master_port=59566 --use_env train.py " \
              f"models/iter_mask/piclick_base448_cocolvis_itermask_{i}m.py " \
              f"--batch-size=136 " \
              f"--ngpus=8 " \
              f" | tee logs/piclick_base448_cocolvis_itermask_{i}m.log"
        os.system(cmd)


if __name__ == '__main__':
    main()
