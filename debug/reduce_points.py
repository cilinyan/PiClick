import random
from functools import partial
from collections import defaultdict
import numpy as np


def points_reduce(points: list, remain_one: bool, base_prob: float = 0.7):
    points_available = [p for p in points if p[2] != -1]
    num_available = len(points_available)
    weights = [base_prob ** i for i in range(num_available + int(not remain_one))]
    sum_weights = sum(weights)
    weights = [w / sum_weights for w in weights]
    num_choice = np.random.choice(
        list(range(int(remain_one), num_available + 1)), size=1, replace=False, p=weights
    )[0]
    points_choices = random.choices(points_available, k=num_choice)
    return points_choices + [(-1, -1, -1)] * (len(points) - len(points_choices))


def static(points, remain_one: bool, keep_prob: float = 0.7, num_sample: int = 1000):
    data = defaultdict(int)
    for _ in range(num_sample):
        ps = points_reduce(points, remain_one, keep_prob)
        n = len([p for p in ps if p[2] != -1])
        data[n] += 1
    data = sorted([(k, v / num_sample) for k, v in data.items()])
    return data


def main():
    points = [(1, 2, 3)] * 9 + [(-1, -1, -1)] * 9
    print('remain')
    print(static(points, remain_one=True, keep_prob=.5, num_sample=1000))
    print('no remain')
    print(static(points, remain_one=False, keep_prob=.5, num_sample=1000))


if __name__ == '__main__':
    main()
