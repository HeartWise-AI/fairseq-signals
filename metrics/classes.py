from __future__ import annotations  # Enables forward references
from typing import Any, List

import pandas as pd
import numpy as np
import os
import pickle


from metrics.utils import (
    TASKS,
    MODELS,
    DATASETS,
    SINGLETON,
    DEVICE,
    N_PERM,
    RATIO,
    set_DEVICE,
    set_N_PERM,
    set_RATIO,
    get_path,
    format_ci_metrics,
    calculate_permuted_metrics,
    calculate_permuted_regression_metrics,
    ci_scores_filename,
    read_manifest,
    get_pred_labels,
    get_centers_pred_labels,
    delong_roc_test,
)

ROOT_DIR='vol'

class AbstractModel:
    def __init__(self, key: str, ds_folder: str, results_folder: str, root_dir=ROOT_DIR, e2e=False):
        self.key = key
        self.ds_folder = ds_folder
        self.results_folder = results_folder
        self.root_dir = root_dir
        self.e2e = e2e
    def get_results_folder(self,  task: AbstractTask) -> str:
        pass
    def get_ds_folder(self, task: AbstractTask):
        return os.path.join(self.ds_folder, task.key)
    
    
        
class DeepECGSL(AbstractModel):
    def __init__(self, key='sl', ds_folder='/media/data1/anolin/for_achille_ssl/for_table', results_folder='/media/data1/anolin/for_achille_ssl/for_table', e2e=False):
        super().__init__(key, ds_folder, results_folder, e2e)
    def get_results_folder(self, task: AbstractTask):
        return os.path.join(self.results_folder, task.key)

class DeepECGSSL(AbstractModel):
    def __init__(self, key='ssl', ds_folder='/media/data1/achilsowa/datasets/fairseq/mhi-mimic-code15/manifest/finetune', root_dir=ROOT_DIR, e2e=False):
        super().__init__(key, ds_folder, '', root_dir, e2e)
    def get_results_folder(self, task: AbstractTask):
        return get_path(task.results_key, root_dir=self.root_dir)
    
class AbstractTask:
    def __init__(self, key, ds:AbstractDataset, model: AbstractModel, results_key: str, type='classification', percentage=100):        
        if percentage != 100:
            key = f'{key}_{percentage}'
            results_key = results_key.split('-')
            results_key = '-'.join(results_key[:-1] + [str(percentage), results_key[-1]])
        if model.e2e:
            results_key = results_key.replace('last-ft', 'e2e-e2e').replace('afib-v2-5', 'afib_5').replace('fevg-v2-40', 'lvef_40').replace('fevg-v2-50', 'lvef_50').replace('lqts-type', 'lqts_type')
        self.key = key
        self.type = type
        self.metrics = ['auroc', 'f1score', 'auprc'] if type == 'classification' else ['mse', 'mae', 'r2', 'pearson_correlation']
        self.labels = []
        self.categories = {}
        self.ds = ds
        self.model = model
        self.percentage = percentage
        self.results_key = results_key
        self.results_folder = ds.get_results_folder(self)

    def get_scores(self, force_evaluation=False, save=True, num_permutations=N_PERM, sample_ratio=RATIO):
        results_path = self.ds.get_results_path(self)
        
        if not force_evaluation and os.path.exists(results_path):
            return pd.read_csv(results_path)
        
        y_pred, y_true = self.ds.get_y(self)
        if self.type == 'classification':
            scores = calculate_permuted_metrics(
                y_true=y_true, 
                y_pred=y_pred, 
                device=DEVICE, 
                y_labels=self.labels, 
                categories=self.categories, 
                num_permutations=num_permutations,
                sample_ratio=sample_ratio
            )    
        else:
            scores = calculate_permuted_regression_metrics( 
                y_true=y_true, 
                y_pred=y_pred, 
                device=DEVICE, 
                y_labels=self.labels, 
                categories=self.categories, 
                num_permutations=num_permutations,
                sample_ratio=sample_ratio
            )

        fscores = format_ci_metrics(scores, self.categories, self.labels, self.metrics)
        if save:
            print(f'results saved in: {results_path}')
            fscores.to_csv(results_path)
            
        return fscores
    
    def delong_test(self, ref:AbstractTask, p_value_col='p_value', ):
        assert self.labels == ref.labels, "task and ref should have same labels"
        assert self.categories == ref.categories, "task and ref should have same categories"
        scores = self.get_scores()
        results_path = self.ds.get_results_path(self)
        y_pred, y_true = self.ds.get_y(self)
        ref_y_pred, ref_y_true = ref.ds.get_y(ref)
        assert (y_true == ref_y_true).all(), "task and ref should have same ground truth"
        p_values = {}
        scores_p_values = []
        for i, label in enumerate(self.labels):
            Yt = y_true[:, i]
            Yp = y_pred[:, i]
            Yrp = ref_y_pred[:, i]
            p_values[f'{label}_auroc'] = 10 ** delong_roc_test(Yt, Yrp, Yp)[0, 0]

        for key, labels in self.categories.items():
            idx = [self.labels.index(label) for label in labels]
            Yt = np.concatenate([y_true[:, i] for i in idx])
            Yp =  np.concatenate([y_pred[:, i] for i in idx])
            Yrp = np.concatenate([ref_y_pred[:, i] for i in idx])
            p_values[f'{key}_micro_auroc'] =  10 ** delong_roc_test(Yt, Yrp, Yp)[0, 0]

        for i in range(len(scores)):
            scores_p_values += [p_values.get(f"{scores.loc[i, 'Category']}_{scores.loc[i, 'Metrics']}", -1)]
        scores[p_value_col] = scores_p_values
        # Drop columns with "Unnamed" in their names
        scores = scores.loc[:, ~scores.columns.str.contains('^Unnamed')]
        scores.to_csv(results_path, index=False)
        return scores
    
    def getDatasetTask(self, ds: AbstractDataset):
        "Same task but on a different dataset"
        return getSingletonTask(self.key, self.model.key, ds.key)

    def getModelTask(self, model: AbstractModel):
        "Same task but on a different model"
        return getSingletonTask(self.key, model.key, self.ds.key)
        
class Labels77Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, percentage=100, e2e=False):
        key = 'labels_77' if percentage == 100 else f'labels_77_{percentage}'
        result_key = 'last-ft-labels-77-bce' if percentage == 100 else f'last-ft-labels-77-{percentage}-bce'
        if e2e:
            result_key = 'last-e2e-labels-77-bce'
        super().__init__(key, ds, model, result_key)
        self.labels =  ['Acute pericarditis', 'QS complex in V1-V2-V3', 'T wave inversion (anterior - V3-V4)', 'Right atrial enlargement','2nd degree AV block - mobitz 1','Left posterior fascicular block','Wolff-Parkinson-White (Pre-excitation syndrome)','Junctional rhythm','Premature ventricular complex',"rSR' in V1-V2",'Right superior axis','ST elevation (inferior - II, III, aVF)','Afib','ST elevation (anterior - V3-V4)','RV1 + SV6 > 11 mm','Sinusal','Monomorph','Delta wave','R/S ratio in V1-V2 >1','Third Degree AV Block','LV pacing','Nonspecific intraventricular conduction delay','ST depression (inferior - II, III, aVF)','Regular','Premature atrial complex','2nd degree AV block - mobitz 2','Left anterior fascicular block','Q wave (septal- V1-V2)','Prolonged QT','Left axis deviation','Left ventricular hypertrophy','ST depression (septal- V1-V2)','Supraventricular tachycardia','Atrial paced','Q wave (inferior - II, III, aVF)','no_qrs','T wave inversion (lateral -I, aVL, V5-V6)','Right bundle branch block','ST elevation (septal - V1-V2)','SV1 + RV5 or RV6 > 35 mm','Right axis deviation','RaVL > 11 mm','Polymorph','Ventricular tachycardia','QRS complex negative in III','ST depression (lateral - I, avL, V5-V6)','1st degree AV block','Lead misplacement','Q wave (posterior - V7-V9)','Atrial flutter','Ventricular paced','ST elevation (posterior - V7-V8-V9)','Ectopic atrial rhythm (< 100 BPM)','Early repolarization','Ventricular Rhythm','Irregularly irregular','Atrial tachycardia (>= 100 BPM)','R complex in V5-V6','ST elevation (lateral - I, aVL, V5-V6)','Brugada','Bi-atrial enlargement','Q wave (lateral- I, aVL, V5-V6)','ST upslopping','T wave inversion (inferior - II, III, aVF)','Regularly irregular','Bradycardia','qRS in V5-V6-I, aVL','Q wave (anterior - V3-V4)','Acute MI','ST depression (anterior - V3-V4)','Right ventricular hypertrophy','T wave inversion (septal- V1-V2)','ST downslopping','Left bundle branch block','Low voltage','U wave','Left atrial enlargement']
        self.categories = {
            "RHYTHM": ['Ventricular tachycardia','Bradycardia','Brugada','Wolff-Parkinson-White (Pre-excitation syndrome)','Atrial flutter','Ectopic atrial rhythm (< 100 BPM)','Atrial tachycardia (>= 100 BPM)','Sinusal','Ventricular Rhythm','Supraventricular tachycardia','Junctional rhythm','Regular','Regularly irregular','Irregularly irregular','Afib','Premature ventricular complex','Premature atrial complex'],
            "CONDUCTION": ['Left anterior fascicular block','Delta wave','2nd degree AV block - mobitz 2','Left bundle branch block','Right bundle branch block','Left axis deviation','Atrial paced','Right axis deviation','Left posterior fascicular block','1st degree AV block','Right superior axis','Nonspecific intraventricular conduction delay','Third Degree AV Block','2nd degree AV block - mobitz 1','Prolonged QT','U wave','LV pacing','Ventricular paced'],
            "CHAMBER ENLARGEMENT": ['Bi-atrial enlargement','Left atrial enlargement','Right atrial enlargement','Left ventricular hypertrophy','Right ventricular hypertrophy'],
            "PERICARDITIS": ['Acute pericarditis'],
            'INFARCT, ISCHEMIA': ['Q wave (septal- V1-V2)','ST elevation (anterior - V3-V4)','Q wave (posterior - V7-V9)','Q wave (inferior - II, III, aVF)','Q wave (anterior - V3-V4)','ST elevation (lateral - I, aVL, V5-V6)','Q wave (lateral- I, aVL, V5-V6)','ST depression (lateral - I, avL, V5-V6)','Acute MI','ST elevation (septal - V1-V2)','ST elevation (inferior - II, III, aVF)','ST elevation (posterior - V7-V8-V9)','ST depression (inferior - II, III, aVF)','ST depression (anterior - V3-V4)'],
            "OTHER": ['ST downslopping','ST depression (septal- V1-V2)','R/S ratio in V1-V2 >1','RV1 + SV6 > 11 mm','Polymorph',"rSR' in V1-V2",'QRS complex negative in III','qRS in V5-V6-I, aVL','QS complex in V1-V2-V3','R complex in V5-V6','RaVL > 11 mm','T wave inversion (septal- V1-V2)','SV1 + RV5 or RV6 > 35 mm','T wave inversion (inferior - II, III, aVF)','Monomorph','T wave inversion (anterior - V3-V4)','T wave inversion (lateral -I, aVL, V5-V6)','Low voltage','Lead misplacement','ST depression (anterior - V3-V4)','Early repolarization','ST upslopping','no_qrs'],
            "ALL": self.labels
        }   
        self.percentage = percentage
        self.title = 'ECG interpretation'

    def update_nlabels(self, n_labels: str):
        """
        Update the task with a new number of labels.
        This is used to create tasks with different numbers of labels.
        """
        self.key = self.key.replace('77', n_labels)
        self.results_key = self.results_key.replace('77', n_labels)
        
        
class Labels47Task(Labels77Task):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, percentage=100):
        super().__init__(ds, model, percentage)
        self.update_nlabels('47')
        
        self.labels = ['Acute pericarditis','Right atrial enlargement','2nd degree AV block - mobitz 1','Left posterior fascicular block','Wolff-Parkinson-White (Pre-excitation syndrome)','Junctional rhythm','Premature ventricular complex',"rSR' in V1-V2",'Right superior axis','ST elevation (inferior - II, III, aVF)','Afib','ST elevation (anterior - V3-V4)','Sinusal','Third Degree AV Block','LV pacing','Nonspecific intraventricular conduction delay','Premature atrial complex','2nd degree AV block - mobitz 2','Left anterior fascicular block','Q wave (septal- V1-V2)','Prolonged QT','Left axis deviation','Left ventricular hypertrophy','Supraventricular tachycardia','Atrial paced','Q wave (inferior - II, III, aVF)','Right bundle branch block','Right axis deviation','Ventricular tachycardia','1st degree AV block','Q wave (posterior - V7-V9)','Atrial flutter','Ventricular paced','Ectopic atrial rhythm (< 100 BPM)','Early repolarization','Ventricular Rhythm','Atrial tachycardia (>= 100 BPM)','ST elevation (lateral - I, aVL, V5-V6)','Bi-atrial enlargement','Q wave (lateral- I, aVL, V5-V6)','Bradycardia','Q wave (anterior - V3-V4)','Acute MI','Right ventricular hypertrophy','Left bundle branch block','Low voltage','Left atrial enlargement']
        
        self.categories = {
            'RHYTHM': ['Ventricular tachycardia','Bradycardia','Wolff-Parkinson-White (Pre-excitation syndrome)','Atrial flutter','Ectopic atrial rhythm (< 100 BPM)','Atrial tachycardia (>= 100 BPM)','Sinusal','Ventricular Rhythm','Supraventricular tachycardia','Junctional rhythm','Afib','Premature ventricular complex','Premature atrial complex'],
            'CONDUCTION': ['Left anterior fascicular block','2nd degree AV block - mobitz 2','Left bundle branch block','Right bundle branch block','Left axis deviation','Atrial paced','Right axis deviation','Left posterior fascicular block','1st degree AV block','Right superior axis','Nonspecific intraventricular conduction delay','Third Degree AV Block','2nd degree AV block - mobitz 1','Prolonged QT','LV pacing','Ventricular paced'],
            'CHAMBER ENLARGEMENT': ['Bi-atrial enlargement','Left atrial enlargement','Right atrial enlargement','Left ventricular hypertrophy','Right ventricular hypertrophy'],
            'PERICARDITIS': ['Acute pericarditis'],
            'INFARCT, ISCHEMIA': ['Q wave (septal- V1-V2)','ST elevation (anterior - V3-V4)','Q wave (posterior - V7-V9)','Q wave (inferior - II, III, aVF)','Q wave (anterior - V3-V4)','ST elevation (lateral - I, aVL, V5-V6)','Q wave (lateral- I, aVL, V5-V6)','Acute MI','ST elevation (inferior - II, III, aVF)'],
            'OTHER': ['Low voltage', 'Early repolarization'],
            'LABELS_14':  ["Wolff-Parkinson-White (Pre-excitation syndrome)", "Premature ventricular complex", "Afib", "Sinusal", "Supraventricular tachycardia", "Right bundle branch block", "Ventricular tachycardia", "1st degree AV block", "Atrial flutter", "Ventricular paced", "Atrial tachycardia (>= 100 BPM)", "Bradycardia", "Acute MI", "Left bundle branch block"],
            'ALL': self.labels,
        }
        

class Labels14Task(Labels77Task):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, percentage=100):
        super().__init__(ds, model, percentage)
        self.update_nlabels('14')
        
        self.labels = ["Wolff-Parkinson-White (Pre-excitation syndrome)", "Premature ventricular complex", "Afib", "Sinusal", "Supraventricular tachycardia", "Right bundle branch block", "Ventricular tachycardia", "1st degree AV block", "Atrial flutter", "Ventricular paced", "Atrial tachycardia (>= 100 BPM)", "Bradycardia", "Acute MI", "Left bundle branch block"]
        self.categories = {
            "RHYTHM": ["Ventricular tachycardia", "Bradycardia", "Wolff-Parkinson-White (Pre-excitation syndrome)", "Atrial flutter", "Atrial tachycardia (>= 100 BPM)", "Sinusal", "Supraventricular tachycardia", "Afib", "Premature ventricular complex"], 
            "CONDUCTION": ["Left bundle branch block", "Right bundle branch block", "1st degree AV block", "Ventricular paced"], 
            # "CHAMBER ENLARGEMENT": [], 
            # "PERICARDITIS": [], 
            "INFARCT, ISCHEMIA": ["Acute MI"], 
            # "OTHER": [], 
            "ALL": self.labels,
        }
        

class Afib5Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, e2e=False):
        super().__init__('afib_5', ds, model, f'last-{"e2e" if e2e else "ft"}-afib-v2-5-bf')
        self.labels = ['label_5y']
        self.title = 'iAF5'

class Lvef40Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, e2e=False):
        super().__init__('lvef_40', ds, model, f'last-{"e2e" if e2e else "ft"}-fevg-v2-40-bce')
        self.labels = ['LVEF_EQUAL_OR_UNDER_40_tte_lvef']
        self.title = 'LVEF ≤ 40'

class Lvef50Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, e2e=False):
        super().__init__('lvef_50', ds, model, f'last-{"e2e" if e2e else "ft"}-fevg-v2-50-bce')
        self.labels =   ['LVEF_UNDER_50_tte_lvef']
        self.title = 'LVEF < 50'

class LvefRegTask(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, e2e=False):
        super().__init__('lvef_reg', ds, model, 'last-ft-fevg-reg-mse', type='regression')
        ['Visually Estimated EF_tte_lvef']
        self.title = 'LVEF Regression'

class LqtsTask(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, e2e=False):
        super().__init__('lqts', ds, model, f'last-{"e2e" if e2e else "ft"}-lqts-bf')
        self.labels = ['LQTS']
        self.title = 'LQTS'

class LqtsTypeTask(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, e2e=False):
        super().__init__('lqts_type', ds, model, f'last-{"e2e" if e2e else "ft"}-lqts-type-bf')
        self.labels = ['LQTS_TYPE_1']
        self.title = 'LQTS Type'

class Genetic2Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, task_key):
        super().__init__(task_key, ds, model, f'last-ft-{task_key}-bce')
        self.labels = ['EUR']
        self.title = 'EUR vs NON EUR'

class RiskTask(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('risk', ds, model, f'last-ft-risk-bce')
        self.labels = ['RISK']
        self.title = 'RISK'

class Risk2Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('risk-2', ds, model, f'last-ft-risk-2-bce')
        self.labels = ['RISK']
        self.title = 'RISK'

class SHDTask(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel, task_key):
        super().__init__(task_key, ds, model, f'last-ft-{task_key}-bce')
        self.labels = ['lvef_lte_45', 'lvwt_gte_13', 'aortic_stenosis_moderate_severe', 'aortic_regurgitation_moderate_severe',
                       'mitral_regurgitation_moderate_severe', 'tricuspid_regurgitation_moderate_severe', 'pulmonary_regurgitation_moderate_severe',
                       'rv_systolic_dysfunction_moderate_severe', 'pericardial_effusion_moderate_large', 'pasp_gte_45','tr_max_gte_32','shd']
        self.title = 'SHD'

class AbstractDataset:
    def __init__(self, key: str, title: str, tasks: List[str]):
        self.tasks = tasks
        self.title = title
        self.key = key
        self.results_folder=''

    def get_results_folder(self, task: AbstractTask):
        pass
    def get_results_path(self, task: AbstractTask):
        pass
    def get_y(self, task: AbstractTask):
        pass
    def get_stats(self, task: AbstractTask, force_evaluation=False, save=True):
        ds_folder = task.model.get_ds_folder(task)
        stats_path = os.path.join(ds_folder, f'stats_{task.ds.key}.pkl')
        if not force_evaluation and os.path.exists(stats_path):
            with open(stats_path, 'rb') as file:
                return pickle.load(file)
            
        _, Y = self.get_y(task)
        stats = {'$count$': Y.shape[0]}
        Y_n = np.sum(Y, axis=0)
        for i, label in enumerate(task.labels):
            stats[label] = Y_n[i]
        for key, cat in task.categories.items():
            idx = [task.labels.index(i) for i in cat]
            Y_n = Y[:, idx]
            stats[key] = np.sum(np.max(Y_n, axis=1), axis=0)
        if save:
            print(f'results saved in: {stats_path}')
            with open(stats_path, 'wb') as file:
                pickle.dump(stats, file)
        return stats

class PublicDataset(AbstractDataset):
    def __init__(self, key: str, title: str, tasks: List[str]):
        super().__init__(key, title, tasks)
    def get_results_folder(self, task):
        return task.model.get_results_folder(task)
    def get_results_path(self, task: AbstractTask):
        results_folder = task.model.get_results_folder(task)
        return os.path.join(results_folder, ci_scores_filename(self.key))    
    def get_y(self, task: AbstractTask):
        tsv_folder = task.model.get_ds_folder(task)
        results_folder = task.model.get_results_folder(task)
        y_pred_pkl, y_pred_path, y_pred_idx, df_path, y_path, y_idx, y_labels = read_manifest(tsv_folder, results_folder, task.ds.key)
        return get_pred_labels(
            header_pkl=y_pred_pkl,
            header_npy=y_pred_path,
            df_path=df_path,
            npy_path=y_path,
            y_idx=y_idx,
            y_pred_idx=y_pred_idx,
            y_labels=y_labels
        )
    
class CenterDataset(AbstractDataset):
    def __init__(self, key: str, title: str, tasks: List[str]):
        super().__init__(key, title, tasks)
        self.results_folder = f'/media/data1/achilsowa/results/fairseq/centers/{key}'
    def get_y(self, task: AbstractTask):
        return get_centers_pred_labels(self.key, task.model.key, task.key) 
    def get_results_path(self, task:AbstractTask):
        return os.path.join(self.results_folder, ci_scores_filename(f'{task.model.key}-{task.key}'))        
    def get_results_folder(self, task):
        return self.results_folder
    
class ConcatDataset(AbstractDataset):
    def __init__(self, key, title, tasks: List[str], ds_list: List[AbstractDataset]):
        super().__init__(key, title, tasks)
        self.results_folder = f'/media/data1/achilsowa/results/fairseq/centers/{key}'
        self.ds_list = ds_list

    def get_y(self, task: AbstractTask):
        y_pred, y_true = [], []
        for ds in self.ds_list:
            ltask = task.getDatasetTask(ds)
            local_pred, local_true = ds.get_y(ltask)
            y_pred += [local_pred]
            y_true += [local_true]
        
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        return y_pred, y_true
    def get_results_path(self, task:AbstractTask):
        return os.path.join(self.results_folder, ci_scores_filename(f'{task.model.key}-{task.key}'))        
    def get_results_folder(self, task):
        return self.results_folder
        
        
def getSingletonModel(type: str, root_dir=ROOT_DIR) -> AbstractModel:
    assert type in MODELS
    key = f'model_{type}'
    if key in SINGLETON: 
        return SINGLETON[key]
    elif type == 'sl':
        model = SINGLETON[key] = DeepECGSL()
        return model
    elif type == 'ssl':
        model = SINGLETON[key] = DeepECGSSL(root_dir=root_dir)
        return model
    elif type == 'e2e-sl':
        folder = '/media/data1/anolin/for_achille_ssl/for_table/e2e'
        model = SINGLETON[key] = DeepECGSL(key='e2e-sl', ds_folder=folder, results_folder=folder, e2e=True)
        return model
    elif type == 'e2e-ssl':
        model = SINGLETON[key] = DeepECGSSL(root_dir=root_dir, e2e=True)
        return model
    elif type == 'ecg-founder':
        folder = '/media/data1/achilsowa/ECGFounder'
        model = SINGLETON[key] = DeepECGSL(key='ecg-founder', ds_folder=folder, results_folder=folder)
        return model
    elif type == 'ecg-fm':
        folder = '/media/data1/achilsowa/ECG-FM'
        model = SINGLETON[key] = DeepECGSL(key='ecg-fm', ds_folder=folder, results_folder=folder)
        return model
    

def getSingletonDataset(subset: str) -> AbstractDataset:
    
    key = f'ds_{subset}'
    if key in SINGLETON:
        return SINGLETON[key]
    elif subset in ['mhi', 'test']:
        ds = SINGLETON[key] = PublicDataset('test', 'MHI (internal)', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50', 'lqts', 'lqts_type'])
        return ds
    elif subset in ['val', 'valid']:
        ds = SINGLETON[key] = PublicDataset('valid', 'MHI (val)', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50', 'lqts', 'lqts_type'])
        return ds
    elif subset in ['mimic', 'mimic_cleaned']:
        ds = SINGLETON[key] = PublicDataset('mimic_cleaned', 'MIMIC-IV', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset in ['ptb', 'ptb_cleaned']:
        ds = SINGLETON[key] = PublicDataset('ptb_cleaned', 'PTB', tasks=['labels_77'])
        return ds
    elif subset in ['clsa', 'clsa_cleaned']:
        ds = SINGLETON[key] = PublicDataset('clsa_cleaned', 'CLSA', tasks=['labels_77'])
        return ds
    elif subset in ['ukb', 'ukbb_cleaned_high_pass_scaled']:
        ds = SINGLETON[key] = PublicDataset('ukbb_cleaned_high_pass_scaled', 'UKB', tasks=['labels_77'])
        return ds
    elif subset in ['external_public', 'epd', 'EPD']:
        ds = SINGLETON[key] = PublicDataset('external_public', 'EPD', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'uw':
        ds = SINGLETON[key] = CenterDataset(subset, 'UW', tasks=['labels_77', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'ucsf':
        ds = SINGLETON[key] = CenterDataset(subset, 'UCSF', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'cshs':
        ds = SINGLETON[key] = CenterDataset(subset, 'CSH', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'hms':
        ds = SINGLETON[key] = CenterDataset(subset, 'MGH', tasks=['labels_77'])
        return ds
    elif subset == 'nyp':
        ds = SINGLETON[key] = CenterDataset(subset, 'NYP', tasks=['labels_77', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'jgh':
        ds = SINGLETON[key] = CenterDataset(subset, 'JGH', tasks=['labels_77', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'chum':
        ds = SINGLETON[key] = CenterDataset(subset, 'CHUM', tasks=['labels_77'])
        return ds
    elif subset in ['external_private', 'ehc', 'EHC']:
        ds = SINGLETON[key] = CenterDataset(subset, 'EHC', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset in ['external', 'ed', 'ED']:
        ds_list = [getSingletonDataset('external_public'), getSingletonDataset('external_private')]
        ds = SINGLETON[key] = ConcatDataset(subset, 'EVD (external)', ds_list=ds_list, tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    else:
        ds = SINGLETON[key] = PublicDataset(subset, subset.upper(), tasks=[])
        return ds
        # PublicDataset('test', 'MHI (internal)'
        # assert subset in DATASETS, f'subset: {subset} not in DATASETS: {DATASETS}'

def getSingletonTask(task_key:str, model_key:str, ds_key: str, percentage=100, root_dir='nas') -> AbstractTask:
    # assert task_key in TASKS
    model = getSingletonModel(model_key, root_dir)
    ds = getSingletonDataset(ds_key)
    assert model is not None, f'model {model_key} not found'
    assert ds is not None, f'dataset {ds_key} not found'
    key = f'task_{task_key}_{ds.key}_{model.key}_{percentage}'

    if key in SINGLETON:
        return SINGLETON[key]
    elif task_key == 'labels_77':
        task = SINGLETON[key] = Labels77Task(ds, model, percentage)
        return task
    elif task_key == 'labels_47':
        task = SINGLETON[key] = Labels47Task(ds, model, percentage)
        return task
    elif task_key == 'labels_14':
        task = SINGLETON[key] = Labels14Task(ds, model, percentage)
        return task
    elif task_key == 'afib_5':
        task = SINGLETON[key] = Afib5Task(ds, model)
        return task
    elif task_key == 'lvef_40':
        task = SINGLETON[key] = Lvef40Task(ds, model)
        return task
    elif task_key == 'lvef_50':
        task = SINGLETON[key] = Lvef50Task(ds, model)
        return task
    elif task_key == 'lvef_reg':
        task = SINGLETON[key] = LvefRegTask(ds, model)
        return task
    elif task_key == 'lqts':
        task = SINGLETON[key] = LqtsTask(ds, model)
        return task
    elif task_key == 'lqts_type':
        task = SINGLETON[key] = LqtsTypeTask(ds, model)
        return task 
    elif task_key == 'risk':
        task = SINGLETON[key] = RiskTask(ds, model)
        return task        
    elif task_key.startswith('genetic') or task_key.startswith('ngenetic') or task_key.startswith('lgenetic'):
        task = SINGLETON[key] = Genetic2Task(ds, model, task_key)
        return task            
    elif task_key in ['shd', 'shd_processed']:
        task = SINGLETON[key] = SHDTask(ds, model, task_key)
        return task            

# Generate Ecg interpretation table
def format_scores(scores):
    Label = 'Category'
    df = scores.copy()

    if 'Label' in df.columns:
        Label = 'Label'
    for i in range(len(df)):
        if df.loc[i, Label] == 'Total':
            df.loc[i, Label] = 'ALL'
    df.index = df[Label] + '_' + df['Metrics']
    #df = df[['Mean', '95% CI']]
    return df

def get_formated_scores(model_key: str, ds_key: str, task_key: str, root_dir='nas', **kwargs):
    task = getSingletonTask(task_key, model_key, ds_key, 100, root_dir)
    return format_scores(task.get_scores(**kwargs))