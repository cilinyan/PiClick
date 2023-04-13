# PYTHONPATH=".":$PYTHONPATH python debug/mask2former_head_check.py
import pdb
import torch
from isegm.model.modeling.mask2former_helper.mask2former_head import Mask2FormerHead

# [256, 512, 1024, 2048]
if __name__ == '__main__':
    head = Mask2FormerHead().cuda()

    feats = [
        torch.rand((4, 256, 56, 56), dtype=torch.float32).cuda(),
        torch.rand((4, 512, 28, 28), dtype=torch.float32).cuda(),
        torch.rand((4, 1024, 14, 14), dtype=torch.float32).cuda(),
        torch.rand((4, 2048, 7, 7), dtype=torch.float32).cuda(),
    ]

    cls_pred_list, mask_pred_list = head(feats)

    pdb.set_trace()
