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
    N_PERM,
    RATIO,
)
from metrics.utils import (
    calculate_permuted_metrics,
    format_ci_metrics,
    ci_scores_filename,
    stable_sigmoid,
)

set_N_PERM(10)
DEVICE = 'cuda:2'
if __name__ == '__main__':
    labels = ['Acute pericarditis', 'QS complex in V1-V2-V3', 'T wave inversion (anterior - V3-V4)', 'Right atrial enlargement','2nd degree AV block - mobitz 1','Left posterior fascicular block','Wolff-Parkinson-White (Pre-excitation syndrome)','Junctional rhythm','Premature ventricular complex',"rSR' in V1-V2",'Right superior axis','ST elevation (inferior - II, III, aVF)','Afib','ST elevation (anterior - V3-V4)','RV1 + SV6 > 11 mm','Sinusal','Monomorph','Delta wave','R/S ratio in V1-V2 >1','Third Degree AV Block','LV pacing','Nonspecific intraventricular conduction delay','ST depression (inferior - II, III, aVF)','Regular','Premature atrial complex','2nd degree AV block - mobitz 2','Left anterior fascicular block','Q wave (septal- V1-V2)','Prolonged QT','Left axis deviation','Left ventricular hypertrophy','ST depression (septal- V1-V2)','Supraventricular tachycardia','Atrial paced','Q wave (inferior - II, III, aVF)','no_qrs','T wave inversion (lateral -I, aVL, V5-V6)','Right bundle branch block','ST elevation (septal - V1-V2)','SV1 + RV5 or RV6 > 35 mm','Right axis deviation','RaVL > 11 mm','Polymorph','Ventricular tachycardia','QRS complex negative in III','ST depression (lateral - I, avL, V5-V6)','1st degree AV block','Lead misplacement','Q wave (posterior - V7-V9)','Atrial flutter','Ventricular paced','ST elevation (posterior - V7-V8-V9)','Ectopic atrial rhythm (< 100 BPM)','Early repolarization','Ventricular Rhythm','Irregularly irregular','Atrial tachycardia (>= 100 BPM)','R complex in V5-V6','ST elevation (lateral - I, aVL, V5-V6)','Brugada','Bi-atrial enlargement','Q wave (lateral- I, aVL, V5-V6)','ST upslopping','T wave inversion (inferior - II, III, aVF)','Regularly irregular','Bradycardia','qRS in V5-V6-I, aVL','Q wave (anterior - V3-V4)','Acute MI','ST depression (anterior - V3-V4)','Right ventricular hypertrophy','T wave inversion (septal- V1-V2)','ST downslopping','Left bundle branch block','Low voltage','U wave','Left atrial enlargement']
    categories = {
        'RHYTHM': ['Ventricular tachycardia','Bradycardia','Brugada','Wolff-Parkinson-White (Pre-excitation syndrome)','Atrial flutter','Ectopic atrial rhythm (< 100 BPM)','Atrial tachycardia (>= 100 BPM)','Sinusal','Ventricular Rhythm','Supraventricular tachycardia','Junctional rhythm','Regular','Regularly irregular','Irregularly irregular','Afib','Premature ventricular complex','Premature atrial complex'],
        'CONDUCTION': ['Left anterior fascicular block','Delta wave','2nd degree AV block - mobitz 2','Left bundle branch block','Right bundle branch block','Left axis deviation','Atrial paced','Right axis deviation','Left posterior fascicular block','1st degree AV block','Right superior axis','Nonspecific intraventricular conduction delay','Third Degree AV Block','2nd degree AV block - mobitz 1','Prolonged QT','U wave','LV pacing','Ventricular paced'],
        'CHAMBER ENLARGEMENT': ['Bi-atrial enlargement','Left atrial enlargement','Right atrial enlargement','Left ventricular hypertrophy','Right ventricular hypertrophy'],
        'PERICARDITIS': ['Acute pericarditis'],
        'INFARCT, ISCHEMIA': ['Q wave (septal- V1-V2)','ST elevation (anterior - V3-V4)','Q wave (posterior - V7-V9)','Q wave (inferior - II, III, aVF)','Q wave (anterior - V3-V4)','ST elevation (lateral - I, aVL, V5-V6)','Q wave (lateral- I, aVL, V5-V6)','ST depression (lateral - I, avL, V5-V6)','Acute MI','ST elevation (septal - V1-V2)','ST elevation (inferior - II, III, aVF)','ST elevation (posterior - V7-V8-V9)','ST depression (inferior - II, III, aVF)','ST depression (anterior - V3-V4)'],
        'OTHER': ['ST downslopping','ST depression (septal- V1-V2)','R/S ratio in V1-V2 >1','RV1 + SV6 > 11 mm','Polymorph',"rSR' in V1-V2",'QRS complex negative in III','qRS in V5-V6-I, aVL','QS complex in V1-V2-V3','R complex in V5-V6','RaVL > 11 mm','T wave inversion (septal- V1-V2)','SV1 + RV5 or RV6 > 35 mm','T wave inversion (inferior - II, III, aVF)','Monomorph','T wave inversion (anterior - V3-V4)','T wave inversion (lateral -I, aVL, V5-V6)','Low voltage','Lead misplacement','ST depression (anterior - V3-V4)','Early repolarization','ST upslopping','no_qrs'],
        'ALL': labels
    }

    folders = [
        #('simclr', '/volume/deepecg/ECGSSL/output/logs/ssl/simclr/ecgresnet_1d50/ecg_classification/ecg77_mhi_test'),
        #('byol', '/volume/deepecg/ECGSSL/output/logs/ssl/byol/ecgresnet_1d50/ecg_classification/ecg77_mhi_test'),
        ('jepa', '/volume/deepecg/ECGSSL/output/logs/ssl/jepa/ecgtb/ecg_classification/ecg77_mhi_test')
    ]
    n_perm = N_PERM
    ratio = RATIO
    for model, folder in folders:
        print('='*5, folder, '='*5)
        y_pred = np.load(os.path.join(folder, 'y_pred.npy'))
        y_true = np.load(os.path.join(folder, 'y_true.npy'))
        y_pred = stable_sigmoid(y_pred)
        scores = calculate_permuted_metrics(
            y_true=y_true, 
            y_pred=y_pred, 
            device=DEVICE, 
            y_labels=labels, 
            categories=categories, 
            num_permutations=n_perm,
            sample_ratio=ratio
        )    

        fscores = format_ci_metrics(scores, categories, labels, metrics=['auroc', 'f1score', 'auprc'])
        filename = ci_scores_filename(f'model-labels_77', n_perm=n_perm, ratio=ratio)
        filename = os.path.join(folder, filename)
        print(f'results saved in: {filename}')
        fscores.to_csv(filename)            
        

    