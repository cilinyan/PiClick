# TRAIN

```shell
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59566 --use_env train.py \
  models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised.py \
  --batch-size=136 \
  --ngpus=8
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59566 --use_env train.py \
  models/iter_mask/multimask_2_base448_cocolvis_itermask.py \
  --batch-size=136 \
  --ngpus=8
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59566 --use_env train.py \
  models/iter_mask/multimask_despair_long_base448_cocolvis_itermask.py \
  --batch-size=136 \
  --ngpus=8
```

# VAL

## Flip + Match

```shell
python scripts/evaluate_model.py NoBRS --gpu=2 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid/006/checkpoints/059.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
```

## Flip + Match + Max_IoU

```shell
conda activate click && python scripts/evaluate_model_max_iou.py NoBRS --gpu=3 \
  --checkpoint=./weights/models/iter_mask/multimask_despair_base448_cocolvis_itermask/003/checkpoints/056.pth \
  --eval-mode=cvpr \
  --datasets=DAVIS \
  --output-tuple --n-clicks 100
```

```shell
conda activate click && python scripts/evaluate_model_max_iou.py NoBRS --gpu=0 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid/006/checkpoints/054.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
```

```shell
conda activate click && python scripts/evaluate_model_max_iou.py NoBRS --gpu=7 \
  --checkpoint=./weights/models/iter_mask/multimask_despair_base448_cocolvis_itermask/003/checkpoints/061.pth \
  --eval-mode=cvpr \
  --datasets=ssTEM,BraTS,OAIZIB \
  --output-tuple --print-ious --save-ious --n-clicks 20
```

## Temp_code

weights/models/iter_mask/multimask_ranking_2mlp_base448_cocolvis_itermask/001/checkpoints/079.pth

```shell
conda activate click && python scripts/evaluate_model_max_iou.py NoBRS --gpu=4 \
  --checkpoint=./weights/models/iter_mask/multimask_despair_base448_cocolvis_itermask/003/checkpoints/056.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
```

```shell
conda activate click && python scripts/evaluate_model_max_iou.py NoBRS --gpu=7 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask_no_ranking_loss/001/checkpoints/064.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
```

```shell
conda activate click && python scripts/evaluate_model_max_iou.py NoBRS --gpu=7 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/015/checkpoints/055.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=5 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/015/checkpoints/055.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley \
  --output-tuple
  
conda activate click && python scripts/evaluate_model_max_iou_448.py NoBRS --gpu=7 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/015/checkpoints/055.pth \
  --eval-mode=cvpr \
  --datasets=DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 --master_port=28171 --use_env train.py \
  models/iter_mask/multimask_despair_base448_cocolvis_itermask.py \
  --batch-size=136 \
  --ngpus=8
python -m torch.distributed.launch --nproc_per_node=8 --master_port=28171 --use_env train.py \
  models/iter_mask/multimask_ranking_2mlp_base448_cocolvis_itermask.py \
  --batch-size=136 \
  --ngpus=8
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59566 --use_env train.py \
  models/iter_mask/multimask_ranking_base448_cocolvis_itermask_v2.py \
  --batch-size=136 \
  --ngpus=8
```

```shell
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=0 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/013/checkpoints/079.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=1 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/013/checkpoints/074.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=2 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/013/checkpoints/069.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=3 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/013/checkpoints/064.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=4 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/013/checkpoints/059.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=5 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/013/checkpoints/054.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=6 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/013/checkpoints/052.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=7 \
  --checkpoint=./weights/models/iter_mask/multimask_ranking_base448_cocolvis_itermask/013/checkpoints/050.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59566 --use_env train.py \
  models/iter_mask/multimask_ranking_base448_cocolvis_itermask.py \
  --batch-size=136 \
  --ngpus=8
```

```shell
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=0 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised/006/checkpoints/050.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=1 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised/006/checkpoints/052.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=2 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised/006/checkpoints/054.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=3 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised/006/checkpoints/059.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=4 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised/006/checkpoints/064.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=5 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised/006/checkpoints/069.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=6 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised/006/checkpoints/074.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple
conda activate click && python scripts/evaluate_model.py NoBRS --gpu=7 \
  --checkpoint=./weights/models/iter_mask/multimask_neck_multi_stage_base448_cocolvis_itermask_mask2former_hybrid_single_mask_supervised/006/checkpoints/079.pth \
  --eval-mode=cvpr \
  --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB \
  --output-tuple

```