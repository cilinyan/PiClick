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
    'RITM-HRNet18': {
        'GrabCut': 'mIoU@1=88.33%; mIoU@2=91.92%; mIoU@3=95.56%; mIoU@4=96.15%; mIoU@5=96.97%; mIoU@6=97.32%; mIoU@7=97.70%; mIoU@8=97.98%; mIoU@9=98.11%; mIoU@10=98.24%;',
        'Berkeley': 'mIoU@1=83.37%; mIoU@2=90.98%; mIoU@3=94.16%; mIoU@4=95.17%; mIoU@5=95.55%; mIoU@6=95.74%; mIoU@7=95.88%; mIoU@8=96.00%; mIoU@9=96.09%; mIoU@10=96.18%;',
        'DAVIS': 'mIoU@1=71.35%; mIoU@2=80.33%; mIoU@3=84.72%; mIoU@4=87.19%; mIoU@5=88.76%; mIoU@6=89.78%; mIoU@7=90.41%; mIoU@8=91.06%; mIoU@9=91.49%; mIoU@10=91.73%;',
        'SBD': 'mIoU@1=71.44%; mIoU@2=81.48%; mIoU@3=85.71%; mIoU@4=87.99%; mIoU@5=89.31%; mIoU@6=90.18%; mIoU@7=90.77%; mIoU@8=91.24%; mIoU@9=91.61%; mIoU@10=91.91%;',
        'COCO_MVal': 'mIoU@1=76.67%; mIoU@2=86.93%; mIoU@3=90.97%; mIoU@4=92.86%; mIoU@5=93.84%; mIoU@6=94.46%; mIoU@7=94.85%; mIoU@8=95.08%; mIoU@9=95.30%; mIoU@10=95.47%;',
        'BraTS': 'mIoU@1=8.55%; mIoU@2=23.49%; mIoU@3=46.38%; mIoU@4=64.47%; mIoU@5=73.37%; mIoU@6=79.35%; mIoU@7=83.07%; mIoU@8=85.48%; mIoU@9=86.71%; mIoU@10=87.86%;',
    },
    'RITM-HRNet32': {
        'GrabCut': 'mIoU@1=87.69%; mIoU@2=91.89%; mIoU@3=94.91%; mIoU@4=95.86%; mIoU@5=97.03%; mIoU@6=97.39%; mIoU@7=97.76%; mIoU@8=97.93%; mIoU@9=98.08%; mIoU@10=98.16%;',
        'Berkeley': 'mIoU@1=83.15%; mIoU@2=91.92%; mIoU@3=94.80%; mIoU@4=95.24%; mIoU@5=95.59%; mIoU@6=95.82%; mIoU@7=96.03%; mIoU@8=96.17%; mIoU@9=96.28%; mIoU@10=96.33%;',
        'DAVIS': 'mIoU@1=72.53%; mIoU@2=81.76%; mIoU@3=85.50%; mIoU@4=88.14%; mIoU@5=89.77%; mIoU@6=90.70%; mIoU@7=91.25%; mIoU@8=91.68%; mIoU@9=91.96%; mIoU@10=92.23%;',
        'SBD': 'mIoU@1=70.61%; mIoU@2=81.87%; mIoU@3=86.33%; mIoU@4=88.47%; mIoU@5=89.80%; mIoU@6=90.60%; mIoU@7=91.18%; mIoU@8=91.62%; mIoU@9=91.93%; mIoU@10=92.18%;',
        'COCO_MVal': 'mIoU@1=76.70%; mIoU@2=87.19%; mIoU@3=91.05%; mIoU@4=92.75%; mIoU@5=93.84%; mIoU@6=94.51%; mIoU@7=94.93%; mIoU@8=95.10%; mIoU@9=95.33%; mIoU@10=95.38%;',
        'BraTS': 'mIoU@1=8.72%; mIoU@2=24.22%; mIoU@3=51.71%; mIoU@4=69.18%; mIoU@5=77.07%; mIoU@6=81.28%; mIoU@7=84.76%; mIoU@8=86.12%; mIoU@9=87.39%; mIoU@10=88.37%;'
    }
}

_DRAW_DATASETS = ['GrabCut', 'Berkeley', 'DAVIS']
_IMG_PER_ROW = 3


def main():
    assert all(d in _DATASETS_ALL for d in _DRAW_DATASETS)
    row = math.ceil(len(_DRAW_DATASETS) / _IMG_PER_ROW)
    fig, axs = plt.subplots(row, _IMG_PER_ROW, gridspec_kw={'height_ratios': [1] * row})
    plt.rc('font', family='Times New Roman')
    for idx, (ax, dataset) in enumerate(zip(axs.flatten(), _DRAW_DATASETS)):
        x_list = range(1, 10)
        miou = fetch_data_by_dataset(dataset, x_list)
        for model, data in miou.items():
            ax.plot(x_list, data, label=model)
        ax.set_title(dataset)
        ax.set_xlabel('Number of Clicks')
        ax.set_ylabel('mIoU Score (%)')
        ax.legend(loc='lower right')
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    pass


def fetch_data_by_dataset(dataset: str, x_list: list):
    res = dict()
    for model, data in _MIOU_RESULTS.items():
        if dataset in data:
            miou_dict = str2dict(data[dataset])
            res[model] = [miou_dict[str(x)] for x in x_list]
    return res


def str2dict(line: str):
    result = re.findall(r'mIoU@(\d{1,2})=(\d{1,2}\.\d{1,2})%', line)
    result = {k: float(v) for k, v in result}
    return result


if __name__ == '__main__':
    main()
