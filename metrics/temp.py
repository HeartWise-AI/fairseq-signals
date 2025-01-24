import pandas as pd
import numpy as np
import torch
import os
import pickle


from metrics.classes import (
    getSingletonDataset,
    getSingletonModel,
    getSingletonTask,
    set_N_PERM,
    DATASETS
)

set_N_PERM(10)
if __name__ == '__main__':
    sl = getSingletonModel('sl')
    datasets = DATASETS
    datasets = ['uw', 'ucsf', 'nyp', 'jgh', 'cshs', 'hms', 'external_private']
    for ds in datasets:
        dataset = getSingletonDataset(ds)
        task = getSingletonTask('labels_77', sl.key, dataset.key)
        print('='*10, f'evaluation of task: dataset: {dataset.key}, {task.key}, model: {sl.key}', '='*10)
        task.get_scores(force_evaluation=True)
        print('\n')

    