import os
import os.path as osp
import matplotlib.pyplot as plt
import re
import math

_DATASETS_ALL = ['GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal', 'ssTEM', 'BraTS', 'OAIZIB', ]

_MIOU_RESULTS = {
    'PiClick-ViT-B': {
        'GrabCut': 'mIoU@1=91.81%; mIoU@2=95.40%; mIoU@3=97.59%; mIoU@4=97.94%; mIoU@5=98.12%; mIoU@6=98.58%; mIoU@7=98.71%; mIoU@8=98.75%; mIoU@9=98.75%; mIoU@10=98.81%;',
        'Berkeley': 'mIoU@1=90.50%; mIoU@2=94.39%; mIoU@3=95.42%; mIoU@4=95.73%; mIoU@5=95.95%; mIoU@6=96.02%; mIoU@7=96.05%; mIoU@8=96.17%; mIoU@9=96.17%; mIoU@10=96.17%;',
        'DAVIS': 'mIoU@1=81.20%; mIoU@2=88.32%; mIoU@3=90.29%; mIoU@4=91.19%; mIoU@5=91.82%; mIoU@6=92.10%; mIoU@7=92.32%; mIoU@8=92.63%; mIoU@9=92.80%; mIoU@10=92.89%;',
        'PascalVOC': 'mIoU@1=83.51%; mIoU@2=92.10%; mIoU@3=94.68%; mIoU@4=95.95%; mIoU@5=96.59%; mIoU@6=97.02%; mIoU@7=97.36%; mIoU@8=97.63%; mIoU@9=97.84%; mIoU@10=98.01%;',
        'SBD': 'mIoU@1=77.51%; mIoU@2=85.15%; mIoU@3=87.71%; mIoU@4=89.05%; mIoU@5=89.89%; mIoU@6=90.51%; mIoU@7=90.91%; mIoU@8=91.20%; mIoU@9=91.47%; mIoU@10=91.66%;',
        'COCO_MVal': 'mIoU@1=84.01%; mIoU@2=90.65%; mIoU@3=92.26%; mIoU@4=93.06%; mIoU@5=93.53%; mIoU@6=93.96%; mIoU@7=94.22%; mIoU@8=94.37%; mIoU@9=94.42%; mIoU@10=94.44%;',
        'ssTEM': 'mIoU@1=39.59%; mIoU@2=81.50%; mIoU@3=88.52%; mIoU@4=90.89%; mIoU@5=92.13%; mIoU@6=92.83%; mIoU@7=93.32%; mIoU@8=93.61%; mIoU@9=93.84%; mIoU@10=93.94%;',
        'BraTS': 'mIoU@1=52.28%; mIoU@2=68.16%; mIoU@3=75.68%; mIoU@4=79.67%; mIoU@5=82.99%; mIoU@6=85.07%; mIoU@7=86.46%; mIoU@8=87.41%; mIoU@9=88.03%; mIoU@10=88.32%;',
        'OAIZIB': 'mIoU@1=25.21%; mIoU@2=47.39%; mIoU@3=57.46%; mIoU@4=63.91%; mIoU@5=68.91%; mIoU@6=72.12%; mIoU@7=73.94%; mIoU@8=75.18%; mIoU@9=76.03%; mIoU@10=76.59%;',
    },
    'RITM-HRNet-18': {
        'GrabCut': 'mIoU@1=88.33%; mIoU@2=91.92%; mIoU@3=95.56%; mIoU@4=96.15%; mIoU@5=96.97%; mIoU@6=97.32%; mIoU@7=97.70%; mIoU@8=97.98%; mIoU@9=98.11%; mIoU@10=98.24%;',
        'Berkeley': 'mIoU@1=83.37%; mIoU@2=90.98%; mIoU@3=94.16%; mIoU@4=95.17%; mIoU@5=95.55%; mIoU@6=95.74%; mIoU@7=95.88%; mIoU@8=96.00%; mIoU@9=96.09%; mIoU@10=96.18%;',
        'DAVIS': 'mIoU@1=71.35%; mIoU@2=80.33%; mIoU@3=84.72%; mIoU@4=87.19%; mIoU@5=88.76%; mIoU@6=89.78%; mIoU@7=90.41%; mIoU@8=91.06%; mIoU@9=91.49%; mIoU@10=91.73%;',
        'PascalVOC': 'mIoU@1=82.40%; mIoU@2=90.23%; mIoU@3=93.49%; mIoU@4=95.06%; mIoU@5=96.19%; mIoU@6=96.85%; mIoU@7=97.36%; mIoU@8=97.69%; mIoU@9=97.89%; mIoU@10=98.11%;',
        'SBD': 'mIoU@1=71.44%; mIoU@2=81.48%; mIoU@3=85.71%; mIoU@4=87.99%; mIoU@5=89.31%; mIoU@6=90.18%; mIoU@7=90.77%; mIoU@8=91.24%; mIoU@9=91.61%; mIoU@10=91.91%;',
        'COCO_MVal': 'mIoU@1=76.67%; mIoU@2=86.93%; mIoU@3=90.97%; mIoU@4=92.86%; mIoU@5=93.84%; mIoU@6=94.46%; mIoU@7=94.85%; mIoU@8=95.08%; mIoU@9=95.30%; mIoU@10=95.47%;',
        'BraTS': 'mIoU@1=8.55%; mIoU@2=23.49%; mIoU@3=46.38%; mIoU@4=64.47%; mIoU@5=73.37%; mIoU@6=79.35%; mIoU@7=83.07%; mIoU@8=85.48%; mIoU@9=86.71%; mIoU@10=87.86%;',
        'ssTEM': 'mIoU@1=34.91%; mIoU@2=70.07%; mIoU@3=84.43%; mIoU@4=89.29%; mIoU@5=90.97%; mIoU@6=92.23%; mIoU@7=92.96%; mIoU@8=93.46%; mIoU@9=93.48%; mIoU@10=93.83%;',
        'OAIZIB': 'mIoU@1=13.48%; mIoU@2=32.60%; mIoU@3=45.46%; mIoU@4=56.59%; mIoU@5=64.31%; mIoU@6=70.00%; mIoU@7=73.96%; mIoU@8=75.57%; mIoU@9=77.55%; mIoU@10=79.15%;',
    },
    'RITM-HRNet-32': {
        'GrabCut': 'mIoU@1=87.69%; mIoU@2=91.89%; mIoU@3=94.91%; mIoU@4=95.86%; mIoU@5=97.03%; mIoU@6=97.39%; mIoU@7=97.76%; mIoU@8=97.93%; mIoU@9=98.08%; mIoU@10=98.16%;',
        'Berkeley': 'mIoU@1=83.15%; mIoU@2=91.92%; mIoU@3=94.80%; mIoU@4=95.24%; mIoU@5=95.59%; mIoU@6=95.82%; mIoU@7=96.03%; mIoU@8=96.17%; mIoU@9=96.28%; mIoU@10=96.33%;',
        'DAVIS': 'mIoU@1=72.53%; mIoU@2=81.76%; mIoU@3=85.50%; mIoU@4=88.14%; mIoU@5=89.77%; mIoU@6=90.70%; mIoU@7=91.25%; mIoU@8=91.68%; mIoU@9=91.96%; mIoU@10=92.23%;',
        'SBD': 'mIoU@1=70.61%; mIoU@2=81.87%; mIoU@3=86.33%; mIoU@4=88.47%; mIoU@5=89.80%; mIoU@6=90.60%; mIoU@7=91.18%; mIoU@8=91.62%; mIoU@9=91.93%; mIoU@10=92.18%;',
        'PascalVOC': 'mIoU@1=82.01%; mIoU@2=90.57%; mIoU@3=93.68%; mIoU@4=95.53%; mIoU@5=96.58%; mIoU@6=97.22%; mIoU@7=97.63%; mIoU@8=97.97%; mIoU@9=98.20%; mIoU@10=98.37%;',
        'COCO_MVal': 'mIoU@1=76.70%; mIoU@2=87.19%; mIoU@3=91.05%; mIoU@4=92.75%; mIoU@5=93.84%; mIoU@6=94.51%; mIoU@7=94.93%; mIoU@8=95.10%; mIoU@9=95.33%; mIoU@10=95.38%;',
        'BraTS': 'mIoU@1=8.72%; mIoU@2=24.22%; mIoU@3=51.71%; mIoU@4=69.18%; mIoU@5=77.07%; mIoU@6=81.28%; mIoU@7=84.76%; mIoU@8=86.12%; mIoU@9=87.39%; mIoU@10=88.37%;',
        'ssTEM': 'mIoU@1=35.25%; mIoU@2=66.49%; mIoU@3=89.26%; mIoU@4=91.18%; mIoU@5=92.49%; mIoU@6=93.30%; mIoU@7=93.12%; mIoU@8=93.71%; mIoU@9=93.80%; mIoU@10=93.93%;',
        'OAIZIB': 'mIoU@1=19.51%; mIoU@2=38.71%; mIoU@3=50.22%; mIoU@4=60.34%; mIoU@5=67.51%; mIoU@6=72.24%; mIoU@7=74.50%; mIoU@8=76.79%; mIoU@9=78.64%; mIoU@10=80.05%;',
    },
    'FocalClick-SegF-B0S1': {
        'GrabCut': 'mIoU@1=76.43%; mIoU@2=87.52%; mIoU@3=91.34%; mIoU@4=93.06%; mIoU@5=94.38%; mIoU@6=96.12%; mIoU@7=97.30%; mIoU@8=97.63%; mIoU@9=97.86%; mIoU@10=97.97%;',
        'Berkeley': 'mIoU@1=72.62%; mIoU@2=85.06%; mIoU@3=89.39%; mIoU@4=91.80%; mIoU@5=93.39%; mIoU@6=94.28%; mIoU@7=94.94%; mIoU@8=95.25%; mIoU@9=95.33%; mIoU@10=95.50%;',
        'COCO_MVal': 'mIoU@1=63.97%; mIoU@2=78.46%; mIoU@3=84.56%; mIoU@4=88.19%; mIoU@5=90.86%; mIoU@6=92.29%; mIoU@7=93.12%; mIoU@8=93.70%; mIoU@9=94.00%; mIoU@10=94.26%;',
        'BraTS': 'mIoU@1=10.04%; mIoU@2=16.49%; mIoU@3=27.27%; mIoU@4=39.46%; mIoU@5=51.62%; mIoU@6=62.22%; mIoU@7=69.23%; mIoU@8=74.66%; mIoU@9=78.99%; mIoU@10=81.52%;',
        'ssTEM': 'mIoU@1=29.39%; mIoU@2=61.82%; mIoU@3=71.96%; mIoU@4=80.91%; mIoU@5=85.60%; mIoU@6=87.88%; mIoU@7=89.31%; mIoU@8=90.49%; mIoU@9=91.47%; mIoU@10=91.89%;',
        'OAIZIB': 'mIoU@1=30.52%; mIoU@2=38.47%; mIoU@3=48.44%; mIoU@4=55.66%; mIoU@5=61.92%; mIoU@6=66.12%; mIoU@7=69.43%; mIoU@8=71.45%; mIoU@9=73.24%; mIoU@10=74.73%;',
        'DAVIS': 'mIoU@1=64.61%; mIoU@2=78.70%; mIoU@3=83.87%; mIoU@4=86.28%; mIoU@5=87.70%; mIoU@6=88.56%; mIoU@7=89.07%; mIoU@8=89.41%; mIoU@9=89.69%; mIoU@10=89.95%;',
        'PascalVOC': 'mIoU@1=61.75%; mIoU@2=78.06%; mIoU@3=84.86%; mIoU@4=88.96%; mIoU@5=91.62%; mIoU@6=93.31%; mIoU@7=94.34%; mIoU@8=95.13%; mIoU@9=95.71%; mIoU@10=96.18%;',
        'SBD': 'mIoU@1=57.45%; mIoU@2=73.06%; mIoU@3=79.51%; mIoU@4=83.67%; mIoU@5=86.42%; mIoU@6=88.07%; mIoU@7=89.06%; mIoU@8=89.74%; mIoU@9=90.29%; mIoU@10=90.72%;'
    },
    'FocalClick-SegF-B0S2': {
        'GrabCut': 'mIoU@1=83.92%; mIoU@2=90.89%; mIoU@3=93.46%; mIoU@4=95.02%; mIoU@5=97.07%; mIoU@6=97.91%; mIoU@7=98.60%; mIoU@8=98.73%; mIoU@9=98.82%; mIoU@10=98.86%;',
        'Berkeley': 'mIoU@1=79.18%; mIoU@2=89.31%; mIoU@3=92.53%; mIoU@4=94.15%; mIoU@5=95.42%; mIoU@6=95.89%; mIoU@7=96.17%; mIoU@8=96.36%; mIoU@9=96.43%; mIoU@10=96.53%;',
        'DAVIS': 'mIoU@1=72.19%; mIoU@2=83.78%; mIoU@3=87.31%; mIoU@4=89.01%; mIoU@5=90.17%; mIoU@6=90.80%; mIoU@7=91.10%; mIoU@8=91.39%; mIoU@9=91.66%; mIoU@10=91.83%;',
        'COCO_MVal': 'mIoU@1=68.85%; mIoU@2=81.94%; mIoU@3=87.71%; mIoU@4=90.72%; mIoU@5=92.66%; mIoU@6=93.68%; mIoU@7=94.20%; mIoU@8=94.55%; mIoU@9=94.80%; mIoU@10=95.00%;',
        'BraTS': 'mIoU@1=11.14%; mIoU@2=24.11%; mIoU@3=40.75%; mIoU@4=55.18%; mIoU@5=67.68%; mIoU@6=75.11%; mIoU@7=79.24%; mIoU@8=82.12%; mIoU@9=84.39%; mIoU@10=85.70%;',
        'ssTEM': 'mIoU@1=25.35%; mIoU@2=58.08%; mIoU@3=78.65%; mIoU@4=84.76%; mIoU@5=86.38%; mIoU@6=88.52%; mIoU@7=89.91%; mIoU@8=91.23%; mIoU@9=91.84%; mIoU@10=92.42%;',
        'OAIZIB': 'mIoU@1=30.93%; mIoU@2=41.94%; mIoU@3=51.16%; mIoU@4=58.43%; mIoU@5=64.06%; mIoU@6=68.21%; mIoU@7=71.27%; mIoU@8=73.70%; mIoU@9=75.53%; mIoU@10=77.00%;',
        'PascalVOC': 'mIoU@1=66.92%; mIoU@2=81.81%; mIoU@3=87.63%; mIoU@4=91.20%; mIoU@5=93.21%; mIoU@6=94.57%; mIoU@7=95.45%; mIoU@8=96.07%; mIoU@9=96.54%; mIoU@10=96.95%;',
        'SBD': 'mIoU@1=63.14%; mIoU@2=76.86%; mIoU@3=82.31%; mIoU@4=85.70%; mIoU@5=87.85%; mIoU@6=89.18%; mIoU@7=90.02%; mIoU@8=90.63%; mIoU@9=91.09%; mIoU@10=91.46%;',
    },
    'FocalClick-SegF-B3S2': {
        'GrabCut': 'mIoU@1=86.97%; mIoU@2=91.90%; mIoU@3=94.10%; mIoU@4=95.42%; mIoU@5=96.98%; mIoU@6=98.70%; mIoU@7=98.93%; mIoU@8=98.99%; mIoU@9=99.06%; mIoU@10=99.09%;',
        'Berkeley': 'mIoU@1=84.18%; mIoU@2=90.53%; mIoU@3=94.27%; mIoU@4=95.51%; mIoU@5=95.93%; mIoU@6=96.68%; mIoU@7=96.83%; mIoU@8=96.94%; mIoU@9=97.01%; mIoU@10=97.05%;',
        'COCO_MVal': 'mIoU@1=75.68%; mIoU@2=86.44%; mIoU@3=90.54%; mIoU@4=92.64%; mIoU@5=93.77%; mIoU@6=94.40%; mIoU@7=94.75%; mIoU@8=95.01%; mIoU@9=95.25%; mIoU@10=95.41%;',
        'BraTS': 'mIoU@1=13.18%; mIoU@2=35.94%; mIoU@3=57.04%; mIoU@4=70.61%; mIoU@5=77.75%; mIoU@6=82.32%; mIoU@7=84.73%; mIoU@8=86.54%; mIoU@9=87.72%; mIoU@10=88.56%;',
        'ssTEM': 'mIoU@1=28.07%; mIoU@2=75.67%; mIoU@3=85.31%; mIoU@4=89.69%; mIoU@5=91.11%; mIoU@6=92.60%; mIoU@7=93.04%; mIoU@8=93.42%; mIoU@9=93.66%; mIoU@10=93.91%;',
        'OAIZIB': 'mIoU@1=28.04%; mIoU@2=47.10%; mIoU@3=55.82%; mIoU@4=63.45%; mIoU@5=67.60%; mIoU@6=71.07%; mIoU@7=73.99%; mIoU@8=76.21%; mIoU@9=78.37%; mIoU@10=79.92%;',
        'DAVIS': 'mIoU@1=76.35%; mIoU@2=84.99%; mIoU@3=88.40%; mIoU@4=90.26%; mIoU@5=91.18%; mIoU@6=91.68%; mIoU@7=92.02%; mIoU@8=92.31%; mIoU@9=92.68%; mIoU@10=92.83%;',
        'PascalVOC': 'mIoU@1=74.44%; mIoU@2=86.28%; mIoU@3=90.82%; mIoU@4=93.26%; mIoU@5=94.80%; mIoU@6=95.74%; mIoU@7=96.46%; mIoU@8=96.98%; mIoU@9=97.34%; mIoU@10=97.62%;',
        'SBD': 'mIoU@1=70.23%; mIoU@2=80.92%; mIoU@3=85.22%; mIoU@4=87.99%; mIoU@5=89.53%; mIoU@6=90.50%; mIoU@7=91.19%; mIoU@8=91.70%; mIoU@9=92.09%; mIoU@10=92.39%;',
    },
    'CDNet-ResNet-34': {
        'GrabCut': 'mIoU@1=89.15%; mIoU@2=91.33%; mIoU@3=94.52%; mIoU@4=95.95%; mIoU@5=97.12%; mIoU@6=98.15%; mIoU@7=97.66%; mIoU@8=98.26%; mIoU@9=97.94%; mIoU@10=97.86%;',
        'Berkeley': 'mIoU@1=85.17%; mIoU@2=87.74%; mIoU@3=92.26%; mIoU@4=93.89%; mIoU@5=94.92%; mIoU@6=95.67%; mIoU@7=95.62%; mIoU@8=95.98%; mIoU@9=96.06%; mIoU@10=96.32%;',
        'COCO_MVal': 'mIoU@1=78.98%; mIoU@2=82.40%; mIoU@3=86.50%; mIoU@4=89.65%; mIoU@5=90.80%; mIoU@6=91.46%; mIoU@7=91.89%; mIoU@8=92.68%; mIoU@9=93.57%; mIoU@10=93.57%;',
        'BraTS': 'mIoU@1=15.21%; mIoU@2=36.51%; mIoU@3=51.70%; mIoU@4=59.34%; mIoU@5=66.37%; mIoU@6=70.87%; mIoU@7=73.50%; mIoU@8=76.10%; mIoU@9=77.81%; mIoU@10=79.71%;',
        'ssTEM': 'mIoU@1=27.75%; mIoU@2=74.50%; mIoU@3=70.70%; mIoU@4=78.34%; mIoU@5=80.61%; mIoU@6=83.22%; mIoU@7=85.60%; mIoU@8=87.76%; mIoU@9=86.70%; mIoU@10=89.31%;',
        'OAIZIB': 'mIoU@1=16.32%; mIoU@2=29.45%; mIoU@3=35.46%; mIoU@4=40.05%; mIoU@5=46.25%; mIoU@6=49.83%; mIoU@7=54.64%; mIoU@8=58.86%; mIoU@9=61.67%; mIoU@10=65.42%;',
        'DAVIS': 'mIoU@1=76.82%; mIoU@2=80.15%; mIoU@3=84.88%; mIoU@4=87.08%; mIoU@5=88.77%; mIoU@6=89.07%; mIoU@7=89.87%; mIoU@8=90.43%; mIoU@9=90.84%; mIoU@10=90.89%;',
        'PascalVOC': 'mIoU@1=77.41%; mIoU@2=80.99%; mIoU@3=86.05%; mIoU@4=89.77%; mIoU@5=91.98%; mIoU@6=93.12%; mIoU@7=94.12%; mIoU@8=94.99%; mIoU@9=95.56%; mIoU@10=95.92%;',
        'SBD': 'mIoU@1=73.37%; mIoU@2=76.21%; mIoU@3=80.55%; mIoU@4=83.77%; mIoU@5=85.83%; mIoU@6=87.19%; mIoU@7=88.15%; mIoU@8=88.96%; mIoU@9=89.61%; mIoU@10=90.01%;',
    },
    'SimpleClick-ViT-B': {
        'GrabCut': 'mIoU@1=88.99%; mIoU@2=94.31%; mIoU@3=95.80%; mIoU@4=97.07%; mIoU@5=97.94%; mIoU@6=98.42%; mIoU@7=98.69%; mIoU@8=98.84%; mIoU@9=98.89%; mIoU@10=98.95%;',
        'Berkeley': 'mIoU@1=84.29%; mIoU@2=92.44%; mIoU@3=94.94%; mIoU@4=96.11%; mIoU@5=96.35%; mIoU@6=96.44%; mIoU@7=96.56%; mIoU@8=96.63%; mIoU@9=96.61%; mIoU@10=96.68%;',
        'DAVIS': 'mIoU@1=72.91%; mIoU@2=83.64%; mIoU@3=87.59%; mIoU@4=89.77%; mIoU@5=90.77%; mIoU@6=91.57%; mIoU@7=92.01%; mIoU@8=92.21%; mIoU@9=92.67%; mIoU@10=92.92%;',
        'PascalVOC': 'mIoU@1=78.94%; mIoU@2=89.17%; mIoU@3=93.18%; mIoU@4=95.25%; mIoU@5=96.33%; mIoU@6=97.08%; mIoU@7=97.64%; mIoU@8=98.02%; mIoU@9=98.26%; mIoU@10=98.47%;',
        'SBD': 'mIoU@1=73.31%; mIoU@2=83.21%; mIoU@3=86.93%; mIoU@4=88.97%; mIoU@5=90.18%; mIoU@6=90.96%; mIoU@7=91.50%; mIoU@8=91.89%; mIoU@9=92.19%; mIoU@10=92.42%;',
        'COCO_MVal': 'mIoU@1=77.46%; mIoU@2=87.76%; mIoU@3=91.91%; mIoU@4=93.39%; mIoU@5=94.18%; mIoU@6=94.71%; mIoU@7=95.03%; mIoU@8=95.28%; mIoU@9=95.45%; mIoU@10=95.56%;',
        'ssTEM': 'mIoU@1=28.48%; mIoU@2=48.31%; mIoU@3=63.24%; mIoU@4=78.48%; mIoU@5=87.22%; mIoU@6=90.45%; mIoU@7=91.82%; mIoU@8=92.69%; mIoU@9=93.31%; mIoU@10=93.73%;',
        'BraTS': 'mIoU@1=9.70%; mIoU@2=22.39%; mIoU@3=43.31%; mIoU@4=61.40%; mIoU@5=70.33%; mIoU@6=76.98%; mIoU@7=81.55%; mIoU@8=84.12%; mIoU@9=85.80%; mIoU@10=87.03%;',
        'OAIZIB': 'mIoU@1=17.28%; mIoU@2=36.40%; mIoU@3=47.50%; mIoU@4=56.57%; mIoU@5=62.03%; mIoU@6=66.55%; mIoU@7=69.82%; mIoU@8=72.45%; mIoU@9=74.52%; mIoU@10=76.01%;',
    },
    'SimpleClick-ViT-L': {
        'GrabCut': 'mIoU@1=87.51%; mIoU@2=95.80%; mIoU@3=96.40%; mIoU@4=97.66%; mIoU@5=98.32%; mIoU@6=98.62%; mIoU@7=98.83%; mIoU@8=98.96%; mIoU@9=98.98%; mIoU@10=98.99%;',
        'Berkeley': 'mIoU@1=82.18%; mIoU@2=93.10%; mIoU@3=95.18%; mIoU@4=95.67%; mIoU@5=96.24%; mIoU@6=96.53%; mIoU@7=96.69%; mIoU@8=96.77%; mIoU@9=96.82%; mIoU@10=96.87%;',
        'DAVIS': 'mIoU@1=74.31%; mIoU@2=84.39%; mIoU@3=88.84%; mIoU@4=90.76%; mIoU@5=91.53%; mIoU@6=91.92%; mIoU@7=92.26%; mIoU@8=92.71%; mIoU@9=92.96%; mIoU@10=93.24%;',
        'PascalVOC': 'mIoU@1=84.06%; mIoU@2=92.18%; mIoU@3=95.10%; mIoU@4=96.52%; mIoU@5=97.27%; mIoU@6=97.78%; mIoU@7=98.10%; mIoU@8=98.43%; mIoU@9=98.62%; mIoU@10=98.85%;',
        'SBD': 'mIoU@1=77.70%; mIoU@2=85.82%; mIoU@3=88.72%; mIoU@4=90.29%; mIoU@5=91.18%; mIoU@6=91.73%; mIoU@7=92.17%; mIoU@8=92.48%; mIoU@9=92.74%; mIoU@10=92.94%;',
        'COCO_MVal': 'mIoU@1=80.62%; mIoU@2=89.29%; mIoU@3=92.60%; mIoU@4=93.89%; mIoU@5=94.57%; mIoU@6=94.99%; mIoU@7=95.32%; mIoU@8=95.50%; mIoU@9=95.62%; mIoU@10=95.75%;',
        'ssTEM': 'mIoU@1=7.66%; mIoU@2=32.74%; mIoU@3=70.29%; mIoU@4=86.60%; mIoU@5=90.13%; mIoU@6=91.90%; mIoU@7=93.02%; mIoU@8=93.61%; mIoU@9=94.05%; mIoU@10=94.31%;',
        'BraTS': 'mIoU@1=12.06%; mIoU@2=39.86%; mIoU@3=60.28%; mIoU@4=70.94%; mIoU@5=77.86%; mIoU@6=82.22%; mIoU@7=84.63%; mIoU@8=86.25%; mIoU@9=87.50%; mIoU@10=88.40%;',
        'OAIZIB': 'mIoU@1=11.48%; mIoU@2=33.10%; mIoU@3=49.09%; mIoU@4=58.73%; mIoU@5=65.43%; mIoU@6=69.21%; mIoU@7=72.17%; mIoU@8=74.12%; mIoU@9=76.26%; mIoU@10=77.26%;',
    },
    'SimpleClick-ViT-H': {
        'GrabCut': 'mIoU@1=86.29%; mIoU@2=94.19%; mIoU@3=96.91%; mIoU@4=97.66%; mIoU@5=98.44%; mIoU@6=98.59%; mIoU@7=98.82%; mIoU@8=98.91%; mIoU@9=98.98%; mIoU@10=99.01%;',
        'Berkeley': 'mIoU@1=83.15%; mIoU@2=92.37%; mIoU@3=95.68%; mIoU@4=96.40%; mIoU@5=96.63%; mIoU@6=96.73%; mIoU@7=96.82%; mIoU@8=96.86%; mIoU@9=96.91%; mIoU@10=96.95%;',
        'DAVIS': 'mIoU@1=73.43%; mIoU@2=83.11%; mIoU@3=87.91%; mIoU@4=90.11%; mIoU@5=91.07%; mIoU@6=91.52%; mIoU@7=92.20%; mIoU@8=92.82%; mIoU@9=93.13%; mIoU@10=93.35%;',
        'PascalVOC': 'mIoU@1=84.76%; mIoU@2=92.11%; mIoU@3=95.02%; mIoU@4=96.20%; mIoU@5=97.03%; mIoU@6=97.58%; mIoU@7=98.02%; mIoU@8=98.34%; mIoU@9=98.56%; mIoU@10=98.67%;',
        'SBD': 'mIoU@1=78.59%; mIoU@2=86.10%; mIoU@3=88.91%; mIoU@4=90.44%; mIoU@5=91.36%; mIoU@6=91.96%; mIoU@7=92.37%; mIoU@8=92.68%; mIoU@9=92.91%; mIoU@10=93.11%;',
        'COCO_MVal': 'mIoU@1=81.01%; mIoU@2=89.40%; mIoU@3=92.77%; mIoU@4=93.98%; mIoU@5=94.66%; mIoU@6=95.16%; mIoU@7=95.40%; mIoU@8=95.60%; mIoU@9=95.76%; mIoU@10=95.88%;',
        'ssTEM': 'mIoU@1=5.66%; mIoU@2=25.80%; mIoU@3=61.66%; mIoU@4=84.13%; mIoU@5=89.30%; mIoU@6=91.26%; mIoU@7=92.58%; mIoU@8=93.47%; mIoU@9=93.87%; mIoU@10=94.10%;',
        'BraTS': 'mIoU@1=17.37%; mIoU@2=46.32%; mIoU@3=65.56%; mIoU@4=74.20%; mIoU@5=79.77%; mIoU@6=83.48%; mIoU@7=85.65%; mIoU@8=86.99%; mIoU@9=88.12%; mIoU@10=88.96%;',
        'OAIZIB': 'mIoU@1=14.44%; mIoU@2=40.52%; mIoU@3=53.00%; mIoU@4=61.36%; mIoU@5=67.39%; mIoU@6=71.03%; mIoU@7=73.34%; mIoU@8=75.40%; mIoU@9=76.71%; mIoU@10=77.58%;',
    }
}


# _COLOR_PARAMS = {
#     'PiClick-ViT-B': dict(color='#F3722C'),
#     'RITM-HRNet-18': dict(marker='#F7D060'),
#     'RITM-HRNet-32': dict(marker='#90BE6D'),
#     'FocalClick-SegF-B0S1': dict(marker='#4D908E'),
#     'FocalClick-SegF-B0S2': dict(marker='#577590'),
#     'FocalClick-SegF-B3S2': dict(marker='#277DA1'),
#     'CDNet-ResNet-34': dict(marker='#43AA8B'),
#     'SimpleClick-ViT-B': dict(marker='#F9844A'),
# }


def draw_curve(
        datasets_draw=('GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal'),
        models_draw=('PiClick-ViT-B', 'RITM-HRNet-18', 'RITM-HRNet-32', 'FocalClick-SegF-B0S1', 'FocalClick-SegF-B0S2',
                     'FocalClick-SegF-B3S2', 'CDNet-ResNet-34', 'SimpleClick-ViT-B'),
        img_per_row=3,
        out_path='vis/converge.pdf',
):
    assert all(d in _DATASETS_ALL for d in datasets_draw)
    row = math.ceil(len(datasets_draw) / img_per_row)
    fig, axs = plt.subplots(row, img_per_row,
                            figsize=(15, 8),
                            # subplot_kw={'aspect': 0.2},
                            subplot_kw={'adjustable': 'box'},
                            gridspec_kw={'height_ratios': [1] * row})
    plt.rc('font', family='Times New Roman')
    for idx, (ax, dataset) in enumerate(zip(axs.flatten(), datasets_draw)):
        x_list = list(range(1, 10))
        miou = fetch_data_by_dataset(dataset, x_list, models_draw)
        for model, data in miou.items():
            if idx == 0:
                ax.plot(x_list, data, label=model, )  # **_DRAW_PARAMS[model]
            else:
                ax.plot(x_list, data, )  # **_DRAW_PARAMS[model]
        ax.set_title(dataset.replace('_', ' '))
        ax.set_xlabel('Number of Clicks')
        ax.set_ylabel('mIoU Score (%)')
        # ax.legend(loc='lower right')
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.subplots_adjust(left=None, bottom=0.13, right=None, top=None, wspace=None, hspace=0.3)
    fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), prop={"size": 10})
    # plt.tight_layout()
    # plt.show()
    plt.savefig(out_path)
    pass


def draw_normal(
        datasets_draw=('GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal'),
        models_draw=('PiClick-ViT-B', 'RITM-HRNet-18', 'RITM-HRNet-32', 'FocalClick-SegF-B0S1', 'FocalClick-SegF-B0S2',
                     'FocalClick-SegF-B3S2', 'CDNet-ResNet-34', 'SimpleClick-ViT-B'),
        out_path='vis/converge.pdf',
):
    assert all(d in _DATASETS_ALL for d in datasets_draw)
    fig, axs = plt.subplots(2, 4,
                            figsize=(22, 8),
                            # subplot_kw={'aspect': 0.2},
                            subplot_kw={'adjustable': 'box'},
                            gridspec_kw={'height_ratios': [1] * 2})
    plt.rc('font', family='Times New Roman')
    # draw barh
    # per models
    labels, values = get_data_per_model(datasets_draw)
    draw_barh_on_ax(axs[0, 0], labels, values,
                    title='(a) PiClick vs. other methods on 6 datasets',
                    xlabel='mIoU delta at 1 click', )
    # per datasets
    labels, values = get_data_per_dataset(datasets_draw)
    row_min = 6
    if len(labels) < row_min:
        labels = list(labels) + [f'tmp_{i}' for i in range(row_min - len(labels))]
        values = list(values) + [0] * (row_min - len(values))
    draw_barh_on_ax(axs[1, 0], labels, values,
                    title='(b) PiClick vs. SimpleClick on 6 datasets',
                    xlabel='mIoU delta at 1 click',
                    skip_value=0)
    # draw curve
    for idx, (ax, dataset) in enumerate(zip(axs[:, 1:].flatten(), datasets_draw)):
        x_list = list(range(1, 10))
        miou = fetch_data_by_dataset(dataset, x_list, models_draw)
        for model, data in miou.items():
            ax.plot(x_list, data, label=model, )  # **_DRAW_PARAMS[model]
            ax.legend(loc='lower right')
        t = ax.set_title(dataset.replace('_', ' '))
        ax.set_xlabel('Number of Clicks')
        ax.set_ylabel('mIoU Score (%)')
        # ax.legend(loc='lower right')
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if idx == len(datasets_draw) - 2:
            ax.set_title('(c) convergence analysis', y=-.27)
    plt.subplots_adjust(left=None, bottom=0.13, right=None, top=None, wspace=0.27, hspace=0.34)
    # fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), prop={"size": 10})
    # plt.tight_layout()
    # plt.show()
    plt.savefig(out_path)
    pass


def draw_medical(
        datasets_draw=('ssTEM', 'BraTS', 'OAIZIB'),
        models_draw=('PiClick-ViT-B', 'RITM-HRNet-18', 'RITM-HRNet-32', 'FocalClick-SegF-B0S1', 'FocalClick-SegF-B0S2',
                     'FocalClick-SegF-B3S2', 'CDNet-ResNet-34', 'SimpleClick-ViT-B'),
        out_path='vis/converge_medical.pdf',
):
    assert all(d in _DATASETS_ALL for d in datasets_draw)
    fig, axs = plt.subplots(2, 2,
                            figsize=(12, 8),
                            # subplot_kw={'aspect': 0.2},
                            subplot_kw={'adjustable': 'box'},
                            gridspec_kw={'height_ratios': [1] * 2})
    plt.rc('font', family='Times New Roman')
    # draw barh
    # per models
    labels, values = get_data_per_model(datasets_draw)
    name_simplified = {
        'PiClick-ViT-B': 'PC-ViT-B',
        'RITM-HRNet-18': 'RITM-H18',
        'RITM-HRNet-32': 'RITM-H32',
        'FocalClick-SegF-B0S1': 'FC-SegF-B0S1',
        'FocalClick-SegF-B0S2': 'FC-SegF-B0S2',
        'FocalClick-SegF-B3S2': 'FC-SegF-B3S2',
        'CDNet-ResNet-34': 'CDN-RN34',
        'SimpleClick-ViT-B': 'SC-ViT-B',
        'SimpleClick-ViT-L': 'SC-ViT-L',
        'SimpleClick-ViT-H': 'SC-ViT-H',
    }
    labels = [name_simplified[d] for d in labels]
    draw_barh_on_ax(axs[0, 0], labels, values,
                    title='(a) PiClick vs. other methods on 3 datasets',
                    xlabel='mIoU delta at 1 click', )
    # draw curve
    title_list = ['(b) mIoU on ssTEM', '(c) mIoU on BraTS', '(d) mIoU on OAIZIB', ]
    for idx, (ax, dataset, title) in enumerate(zip(axs.flatten()[1:], datasets_draw, title_list)):
        x_list = list(range(1, 10))
        miou = fetch_data_by_dataset(dataset, x_list, models_draw)
        for model, data in miou.items():
            ax.plot(x_list, data, label=model, )  # **_DRAW_PARAMS[model]
            ax.legend(loc='lower right')
        ax.set_xlabel('Number of Clicks')
        ax.set_ylabel('mIoU Score (%)')
        # ax.legend(loc='lower right')
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, y=-.27)
    plt.subplots_adjust(left=None, bottom=0.13, right=None, top=None, wspace=0.27, hspace=0.34)
    # fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), prop={"size": 10})
    # plt.tight_layout()
    # plt.show()
    plt.savefig(out_path)


def get_data_per_model(
        datasets_draw=('GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal'),
):
    # 获取每个模型在每个数据集上 1 click 的 平均mIoU
    model_miou = dict()
    for model, model_result in _MIOU_RESULTS.items():
        miou = sum([str2dict(model_result[d])[str(1)] for d in datasets_draw]) / len(datasets_draw)
        model_miou[model] = miou
    miou_piclick = model_miou.pop('PiClick-ViT-B')
    model_miou = sorted(model_miou.items(), key=lambda x: x[1], reverse=True)

    data = dict()
    for model, miou in model_miou:
        data[model] = miou_piclick - miou

    return list(data.keys()), list(data.values())


def draw_bar_per_model(
        datasets_draw=('GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal'),
):
    # 提取数据和标签
    labels, values = get_data_per_model(datasets_draw)
    # 绘图
    fig, ax = plt.subplots()
    draw_barh_on_ax(ax, labels, values,
                    xlabel='mIoU delta at 1 click',
                    # title='PiClick vs. other methods on 6 datasets',
                    )
    # 显示图形
    plt.show()
    pass


def get_data_per_dataset(
        datasets_draw=('GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal'),
):
    miou_piclick = {
        dataset: str2dict(dataset_miou)[str(1)]
        for dataset, dataset_miou in _MIOU_RESULTS['PiClick-ViT-B'].items() if dataset in datasets_draw
    }
    miou_simpleclick = {
        dataset: str2dict(dataset_miou)[str(1)]
        for dataset, dataset_miou in _MIOU_RESULTS['SimpleClick-ViT-B'].items() if dataset in datasets_draw
    }
    miou_diff = {
        dataset: miou_piclick[dataset] - miou_simpleclick[dataset] for dataset in datasets_draw
    }
    miou_diff = sorted(miou_diff.items(), key=lambda x: x[1], reverse=False)
    return list(zip(*miou_diff))


def draw_bar_per_dataset(
        datasets_draw=('GrabCut', 'Berkeley', 'DAVIS', 'PascalVOC', 'SBD', 'COCO_MVal'),
):
    labels, values = get_data_per_dataset(datasets_draw)
    # 绘图
    fig, ax = plt.subplots()
    draw_barh_on_ax(ax, labels, values,
                    # xlabel='mIoU delta at 1 click', title='PiClick vs. SimpleClick on 6 datasets',
                    )

    # 显示图形
    plt.show()


def draw_barh_on_ax(ax, labels, values, xlabel=None, ylabel=None, title=None, skip_value=None):
    labels = [x.replace('_', ' ') for x in labels]
    width = 0.8 if len(labels) > 7 else 0.6
    ax.barh(labels, values, color=['blue' if v <= 0 else '#E86A33' for v in values], zorder=100, height=width)
    # 标注数值
    default_params = dict(color='black', verticalalignment='center')
    for i, v in enumerate(values):
        if skip_value is not None and v == skip_value:
            continue
        if v >= 0:
            ax.text(v + 0.2, i, f'{v:.2f}', horizontalalignment='left', **default_params, zorder=100)
        else:
            ax.text(v - 0.2, i, f'{v:.2f}', horizontalalignment='right', **default_params, zorder=100)
    ax.grid(zorder=0, alpha=0.9, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置轴标签
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        fig = ax.get_figure()
        fig.subplots_adjust(left=0.24, bottom=0.15)  # 增加底部留白
        ax.set_title(title, y=-.27)


def fetch_data_by_dataset(dataset: str, x_list: list, models: list):
    res = dict()
    for model, data in _MIOU_RESULTS.items():
        if model not in models: continue
        if dataset in data:
            miou_dict = str2dict(data[dataset])
            res[model] = [miou_dict[str(x)] for x in x_list]
    return res


def str2dict(line: str):
    result = re.findall(r'mIoU@(\d{1,2})=(\d{1,2}\.\d{1,2})%', line)
    result = {k: float(v) for k, v in result}
    return result


if __name__ == '__main__':
    # draw_curve()
    # draw_bar_per_model()
    # draw_bar_per_dataset()
    draw_normal()
    # draw_medical()
