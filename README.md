## [PiClick: Picking the desired mask in click-based interactive segmentation](https://arxiv.org/abs/2304.11609)

<p align="center">
  <img src="./assets/architecture.png" alt="drawing", width="700"/>
</p>

## Visual Results

<p align="center">
  <img src="./assets/segmentation_results.png" alt="drawing", width="700"/>
</p>

## Environment

Training and evaluation environment: Python3.8.16, PyTorch 1.11.0, CentOS 7.9, CUDA 11.4, mmcv-full 1.6.2. Run the following command to
install required packages.

```
pip3 install -r requirements.txt
```

You also need to configue the paths to the datasets in [config.yml](./config.yml) before training or testing.

## Evaluation

Before evaluation, please download the datasets and models, and then configure the path in [config.yml](./config.yml).

Use the following code to evaluate the base model.

```
# ViT-B
python scripts/evaluate_model.py NoBRS --gpu=0 \
  --checkpoint=./weights/piclick_base448.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,SBD,DAVIS,PascalVOC,COCO_MVal,ssTEM,BraTS,OAIZIB
# ViT-L
python scripts/evaluate_model.py NoBRS --gpu=0 \
  --checkpoint=./weights/piclick_large448.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,SBD,DAVIS,PascalVOC,COCO_MVal,ssTEM,BraTS,OAIZIB
# ViT-H
python scripts/evaluate_model.py NoBRS --gpu=0 \
  --checkpoint=./weights/piclick_huge448.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,SBD,DAVIS,PascalVOC,COCO_MVal,ssTEM,BraTS,OAIZIB
```

## Training

Before training, please download the [MAE](https://github.com/facebookresearch/mae) pretrained weights (click to
download: [ViT-Base](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), [ViT-Large](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth), [ViT-Huge](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth)).

Use the following code to train a base model on COCO+LVIS dataset:

```
# ViT-B
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59566 --use_env train.py \
  models/iter_mask/piclick_base448_cocolvis_itermask.py \
  --batch-size=136 \
  --ngpus=8
# ViT-L
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59516 --use_env train.py \
  models/iter_mask/piclick_large448_cocolvis_itermask.py \
  --batch-size=56 \
  --ngpus=8
# ViT-H (A100, 80G)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59516 --use_env train.py \
  models/iter_mask/piclick_huge448_cocolvis_itermask.py \
  --batch-size=96 \
  --ngpus=8 \
  --workers 48
```

## Download

PiClick models: [Google Drive](https://drive.google.com/drive/folders/1-ZOZ0o-wYGhiHfOzpyqq_ikY6gLLbP8S?usp=sharing), [Baidu Wangpan(1234)](https://pan.baidu.com/s/1T2dr12Y7EbcblVJLDyXn3w)

BraTS dataset (369
cases): [Google Drive](https://drive.google.com/drive/folders/1B6y1nNBnWU09EhxvjaTdp1XGjc1T6wUk?usp=sharing)

OAI-ZIB dataset (150
cases): [Google Drive](https://drive.google.com/drive/folders/1B6y1nNBnWU09EhxvjaTdp1XGjc1T6wUk?usp=sharing)

SBD dataset: [Google Drive](https://drive.google.com/file/d/1LwlHbpDU2Zns48F5YVAaUPo_g-FSxcjx/view?usp=sharing)

Other datasets: [RITM Github](https://github.com/saic-vul/ritm_interactive_segmentation)

## License

The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever
you want as long as you include the original copyright and license notice in any copy of the software/source.

## Note

The according annotation platform will be released upon publication.

## Citation

```
@article{yan2023piclick,
  title={PiClick: Picking the desired mask in click-based interactive segmentation},
  author={Yan, Cilin and Wang, Haochen and Liu, Jie and Jiang, Xiaolong and Hu, Yao and Tang, Xu and Kang, Guoliang and Gavves, Efstratios},
  journal={arXiv preprint arXiv:2304.11609},
  year={2023}
}
```

## Acknowledgement

Our project is developed based on
[RITM](https://github.com/saic-vul/ritm_interactive_segmentation),
[SimpleClick](https://github.com/uncbiag/SimpleClick/tree/v1.0) and
[mmdetection](https://github.com/open-mmlab/mmdetection).
We thank the authors for their great work.
