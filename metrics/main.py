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
    ssl = getSingletonModel('ssl')
    datasets = []
    for ds in datasets:
        dataset = getSingletonDataset(ds)
        for key in dataset.tasks:
            for model in [sl, ssl]:
                task = getSingletonTask(key, model.key, dataset.key)
                print('='*10, f'evaluation of task: dataset: {dataset.key}, {task.key}, model: {model.key}', '='*10)
                task.get_scores(force_evaluation=True)
                print('\n')

    ds = getSingletonDataset('external')
    for m_key in ['sl', 'ssl']:
        for t_key in ds.tasks:
            print('='*10, f'evaluation of task: dataset: {ds.key}, {t_key}, model: {m_key}', '='*10)
            task = getSingletonTask(t_key, m_key, ds.key)
            ds.get_stats(task, True, True)
            task.get_scores(True, True)
