from typing import Any, List
import pandas as pd
import os
import numpy as np
import argparse
from metrics.classes import (
    getSingletonDataset,
    getSingletonModel,
    getSingletonTask,
    SINGLETON,
    DATASETS,
    DEVICE,
    get_formated_scores
)
from metrics.utils import (
    plot_spider,
    plot_histo,
    stable_sigmoid,
    to_percentage,
    sl_ssl_ds_size,
    stable_sigmoid
)
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='which device to use on local machine')
parser.add_argument('--model', type=str, help='which model to evaluate: sl or ssl')
parser.add_argument('--num_perm', type=int, default=1000, help='number of permutations')
parser.add_argument('--percentage', type=int, default=100, help='percentage of trained')

if __name__ == '__main__':
    args = parser.parse_args()
    print (args)
    DEVICE = args.device
    task = getSingletonTask('labels_77', args.model, 'mhi', args.percentage)
    print(task.key)
    score = task.get_scores(force_evaluation=True, save=True, num_permutations=args.num_perm)
    print(score)
    