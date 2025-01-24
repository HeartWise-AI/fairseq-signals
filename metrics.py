from __future__ import annotations  # Enables forward references
from typing import Any, List

import pandas as pd
import numpy as np
import torch
import os
import pickle

from sklearn.metrics import mean_squared_error, r2_score, roc_curve, mean_absolute_error
import torchmetrics.functional.classification as Fc

from scipy.stats import spearmanr, linregress, pearsonr
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm 
import functools



ROOT_DIR_NAS='/media/data1/achilsowa/results/fairseq/outputs/'
ROOT_DIR_VOL='/volume/deepecg/fairseq-signals/outputs/'
ROOT_DIR = 'nas'
EVAL_CI = FORMAT = False
DEVICE = "cuda:2"
N_PERM = 1000
RATIO = 0.7

TASKS = ['labels_77', 'afib_5', 'lvef_40', 'lvef_50', 'lqts', 'lqts_type']
DATASETS = ['mhi', 'mimic', 'ptb', 'ukb', 'clsa', 'external_public', 'uw', 'ucsf', 'jgh', 'nyp', 'hms', 'cshs', 'external_private']
MODELS = ['sl', 'ssl']
SINGLETON = {}

def try_catch_fn(fn, default_val=np.nan):
    try:
        result = fn()
        # Convert tensor to a scalar if applicable
        if isinstance(result, torch.Tensor):
            return result.item()
        return result
    except Exception as e:
        print('Error occurred when evaluating metrics:', e)
        return default_val

def stable_sigmoid(x):
    x_safe = np.clip(x, -500, 500)
    return np.where(x_safe >= 0,
                    1 / (1 + np.exp(-x_safe)),
                    np.exp(x_safe) / (1 + np.exp(x_safe)))

def ci_scores_filename(filename:str, prefix='ci_scores', n_perm=N_PERM, ratio=RATIO):
    return f'{prefix}_{filename}_{ratio}_{n_perm}.csv'

def find_optimal_threshold_youden_index_multilabel(true_labels, pred_scores):
    optimal_thresholds = []
    youden_indices = []

    for i in range(true_labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(true_labels[:, i], pred_scores[:, i])
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds.append(optimal_threshold)
        youden_indices.append(youden_index[optimal_idx])

    return np.array(optimal_thresholds), np.array(youden_indices)

def calculate_permuted_metrics(y_true, y_pred, device, y_labels, categories, num_permutations=100, sample_ratio=0.8, youden_threshold=False):
    permuted_scores = {}

    for perm in tqdm(range(num_permutations), desc="Calculating Permutations"):
        # Sample 80% of data
        indices = np.random.choice(len(y_true), int(len(y_true) * sample_ratio), replace=False)
        
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        
        # Calculate metrics on the sample

        metrics = calculate_metrics(y_true_sample, y_pred_sample, device, y_labels, categories, youden_threshold=youden_threshold)

        # Collect results for each metric
        for metric_key, metric_value in metrics.get("tm", {}).items():
            # Only collect category-level metrics, not individual y_labels
            if any(category in metric_key for category in categories) or any(y_label in metric_key for y_label in y_labels):
                if isinstance(metric_value, (float, int)):
                    permuted_scores.setdefault(metric_key, []).append(metric_value)
                else:
                    print(f"Warning: Metric {metric_key} in permutation {perm} returned a non-numeric value: {metric_value}")

    # Compute mean and 95% CI for each category-level metric only
    category_mean_and_ci = {}
    for metric, values in permuted_scores.items():
        if values:
            mean = np.mean(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            category_mean_and_ci[metric] = {"mean": mean, "95% CI": (ci_lower, ci_upper)}
        else:
            category_mean_and_ci[metric] = {"mean": np.nan, "95% CI": (np.nan, np.nan)}

    return category_mean_and_ci

def calculate_metrics(y_true, y_pred, device, y_labels, categories, single=True, group=True, youden_threshold=False):
    y_pred = np.where(np.isnan(y_pred), 0, y_pred).astype(float)
    y_pred = stable_sigmoid(y_pred)
    
    if youden_threshold:
        optimal_thresholds, _ = find_optimal_threshold_youden_index_multilabel(y_true, y_pred)
        y_pred_thresholded = (y_pred >= optimal_thresholds).astype(int)
    else:
        y_pred_thresholded = (y_pred >= 0.5).astype(int)

    t_y_true = torch.tensor(y_true, device=device, dtype=torch.long)
    t_y_pred = torch.tensor(y_pred, device=device, dtype=torch.float)
    t_y_pred_threshold = torch.tensor(y_pred_thresholded, device=device, dtype=torch.long)
    
    scores = {"tm": {}}
    task = "binary"
    
    if single:
        for id, y_label in enumerate(y_labels):
            t_pred = t_y_pred[:, id]
            t_target = t_y_true[:, id]
            
            scores["tm"][f"{y_label}_auroc"] = try_catch_fn(lambda: Fc.auroc(t_pred, t_target, task=task))
            scores["tm"][f"{y_label}_auprc"] = try_catch_fn(lambda: Fc.average_precision(t_pred, t_target, task=task))
            scores["tm"][f"{y_label}_f1score"] = try_catch_fn(lambda: Fc.f1_score(t_pred, t_target, task=task))
            scores["tm"][f"{y_label}_acc"] = try_catch_fn(lambda: Fc.accuracy(t_pred, t_target, task=task))
            scores["tm"][f"{y_label}_bacc"] = try_catch_fn(lambda: Fc.accuracy(t_y_pred_threshold[:, id], t_target, task='multiclass', num_classes=2, average='macro'))
    
    if group:
        for category, elts in categories.items():
            idx = [y_labels.index(s) for s in elts if s in elts]
            num_labels = len(idx)
            if num_labels == 0: continue
            t_pred = t_y_pred[:, idx]
            t_target = t_y_true[:, idx]
            task = "binary" if num_labels == 1 else "multilabel"
            
            for avg in ["macro", "micro"]:
                scores["tm"][f"{category}_{avg}_auroc"] = try_catch_fn(lambda: Fc.auroc(t_pred, t_target, task=task, num_labels=num_labels, average=avg))
                scores["tm"][f"{category}_{avg}_auprc"] = try_catch_fn(lambda: Fc.average_precision(t_pred, t_target, task=task, num_labels=num_labels, average=avg))
                scores["tm"][f"{category}_{avg}_f1score"] = try_catch_fn(lambda: Fc.f1_score(t_pred, t_target, task=task, num_labels=num_labels, average=avg))
            
            scores["tm"][f"{category}_acc"] = try_catch_fn(lambda: Fc.accuracy(t_y_pred_threshold[:, idx], t_target, task='multiclass', num_classes=2, average='micro'))
            scores["tm"][f"{category}_bacc"] = try_catch_fn(lambda: Fc.accuracy(t_y_pred_threshold[:, idx], t_target, task='multiclass', num_classes=2, average='macro'))
    
    return scores    

def calculate_regression_metrics(y_true, y_pred, device='cuda:0'):
    """
    Calculate regression metrics given model outputs and ground truth labels.

    Parameters:
    outputs (torch.Tensor): The raw output from the model.
    labels (torch.Tensor): The ground truth continuous labels.
    device (str): The device on which to perform calculations (e.g., 'cuda:0').

    Returns:
    dict: A dictionary containing the calculated metrics (MSE, R², Pearson Correlation, MAE).
    """
    #outputs = torch.as_tensor(outputs, device=device, dtype=torch.float32).squeeze()
    #labels = torch.as_tensor(labels, device=device, dtype=torch.float32).squeeze()
    
    # Move data to CPU and convert to 1D numpy arrays for metric calculations
    #outputs_np = outputs.cpu().numpy().ravel()
    #labels_np = labels.cpu().numpy().ravel()

     # Move data to CPU and convert to 1D numpy arrays for metric calculations
    outputs_np = y_pred.ravel()
    labels_np = y_true.ravel()

    # Calculate regression metrics
    mse_value = mean_squared_error(labels_np, outputs_np)
    mae_value = mean_absolute_error(labels_np, outputs_np)
    r2_value = r2_score(labels_np, outputs_np)
    pearson_corr, _ = pearsonr(labels_np, outputs_np)
    
    return {
        'mse': mse_value/100,
        'mae': mae_value/100,
        'r2': r2_value,
        'pearson_correlation': pearson_corr
    }

def bootstrap_regression_metrics(outputs, labels, device='cuda:0', n_iterations=1000, sample_ratio=0.7):
    """
    Perform bootstrapping to obtain mean and 95% CI for regression metrics.

    Parameters:
    outputs (torch.Tensor): The raw output from the model.
    labels (pandas.Series, numpy.ndarray, or torch.Tensor): The ground truth continuous labels.
    device (str): The device on which to perform calculations.
    n_iterations (int): Number of bootstrapping iterations.
    sample_ratio (float): Proportion of data to sample in each iteration.

    Returns:
    dict: Mean and 95% CI for each metric.
    """
    # Convert labels to torch tensor if they are not already
    if isinstance(labels, pd.Series):
        labels = torch.tensor(labels.values, device=device, dtype=torch.float32)
    elif isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, device=device, dtype=torch.float32)
    else:
        labels = labels.to(device).float()

    metrics_results = {'mse': [], 'mae': [], 'r2': [], 'pearson_correlation': []}
    
    for _ in range(n_iterations):
        # Randomly sample 70% of data
        indices = torch.randperm(len(labels))[:int(len(labels) * sample_ratio)]
        sampled_outputs = outputs[indices]
        sampled_labels = labels[indices]
        
        # Calculate metrics for the sampled data
        metrics = calculate_regression_metrics(sampled_outputs, sampled_labels, device=device)
        
        # Store metrics for each iteration
        for metric, value in metrics.items():
            metrics_results[metric].append(value)

    # Compute mean and 95% CI for each metric
    metrics_summary = {}
    for metric, values in metrics_results.items():
        mean_value = np.mean(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        metrics_summary[metric] = {'mean': mean_value, '95% CI': (ci_lower, ci_upper)}

    return metrics_summary

def calculate_regression_metrics2(y_true, y_pred, device='cuda:0'):
    """
    Calculate regression metrics given model outputs and ground truth labels.

    Parameters:
    y_true (torch.Tensor): The raw output from the model.
    y_pred (torch.Tensor): The ground truth continuous labels.
    device (str): The device on which to perform calculations (e.g., 'cuda:0').

    Returns:
    dict: A dictionary containing the calculated metrics (MSE, R², Pearson Correlation).
    """
    
    # Move data to CPU and convert to 1D numpy arrays for metric calculations
    outputs_np = y_pred.ravel()
    labels_np = y_true.ravel()

    # Calculate regression metrics
    mse_value = mean_squared_error(labels_np, outputs_np)
    r2_value = r2_score(labels_np, outputs_np)
    pearson_corr, _ = pearsonr(labels_np, outputs_np)
    
    return {
        'mse': mse_value/100, # because it will be multiplied by 100 for Mean and CI
        'r2': r2_value,
        'pearson_correlation': pearson_corr
    }

def calculate_permuted_regression_metrics(y_true, y_pred, device, y_labels, categories, num_permutations=1000, sample_ratio=0.7):
    """
    Perform bootstrapping to obtain mean and 95% CI for regression metrics.

    Parameters:
    outputs (torch.Tensor): The raw output from the model.
    labels (pandas.Series or torch.Tensor): The ground truth continuous labels.
    device (str): The device on which to perform calculations.
    n_iterations (int): Number of bootstrapping iterations.
    sample_ratio (float): Proportion of data to sample in each iteration.

    Returns:
    dict: Mean and 95% CI for each metric.
    """
    
    metrics_results = {'mse': [], 'mae': [], 'r2': [], 'pearson_correlation': []}
    
    for perm in tqdm(range(num_permutations), desc="Calculating Permutations"):
        # Sample 80% of data
        indices = np.random.choice(len(y_true), int(len(y_true) * sample_ratio), replace=False)
        
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
                
        # Calculate metrics for the sampled data
        metrics = calculate_regression_metrics(y_true_sample, y_pred_sample, device=device)
        
        # Store metrics for each iteration
        for metric, value in metrics.items():
            metrics_results[metric].append(value)

    # Compute mean and 95% CI for each metric
    metrics_summary = {}
    assert len(y_labels) == 1, "Only one label supported"
    for metric, values in metrics_results.items():
        mean_value = np.mean(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        metrics_summary[f'{y_labels[0]}_{metric}'] = {'mean': mean_value, '95% CI': (ci_lower, ci_upper)}
    return metrics_summary

def metrics_headers(y_labels, categories, single=True, group=True):
    headers = []
    if single: 
        for label in y_labels:
            for m in ["auroc", "auprc", "f1score"]:
                headers += [f"{label}_{m}"]

    if group: 
        for category, elts in categories.items():
            idx = [y_labels.index(s) for s in elts if s in elts]
            num_labels = len(idx)
            if num_labels == 0: continue # No items in this category
            for m in ["auroc", "auprc", "f1score"]:
                for avg in ["micro", "macro"]:
                    headers += [f"{category}_{avg}_{m}"]
    return headers

def format_group_metrics(scores, categories, metrics=['acc', 'b-acc', 'auroc', 'f1score', 'auprc']):
    level1 = []
    level2 = []
    level3a = []
    level3b = []

    for category in categories.keys():
        for m in metrics:
            if m in ['acc', 'b-acc']:
                level1 = level1 + [category]        
                level2 = level2 + [m]
                level3a = level3a + [float(scores['tm'][f'{category}_{m}'])]
                level3b = level3b + [float(scores['sk'][f'{category}_{m}'])]
            else:
                for avg in ['micro', 'macro']:
                    level1 = level1 + [category]        
                    level2 = level2 + [f'{avg}_{m}']
                    level3a = level3a + [float(scores['tm'][f'{category}_{avg}_{m}'])]
                    #level3b = level3b + [float(scores['sk'][f'{category}_{avg}_{m}'])]

    df = pd.DataFrame({'Category': level1, 'Metrics': level2, 'Scores (TM)': level3a, }) #'Scores (SK)': level3b})
    return df

def format_ci_metrics(scores, categories, y_labels=[], metrics=['acc', 'auroc', 'f1score', 'auprc'], percentage=True):
    level1 = []
    level2 = []
    level3a = []
    level3b = []

    def to_percentage(num, percentage=True):
        return f'{float(num)*100:.2f}' if percentage else f'{float(num):.3f}'
        
    for category in categories.keys():
        for m in metrics:
            if m in ['acc', 'b-acc']:
                level1 = level1 + [category]        
                level2 = level2 + [m]
                val = to_percentage(scores[f'{category}_{m}']['mean'])
                level3a = level3a + [val]
                val = ' - '.join([str(to_percentage(v, percentage)) for v in scores[f'{category}_{m}']['95% CI']])
                level3b = level3b + [f'[{val}]']
            else:
                for avg in ['micro', 'macro']:
                    level1 = level1 + [category]        
                    level2 = level2 + [f'{avg}_{m}']
                    val = to_percentage(scores[f'{category}_{avg}_{m}']['mean'])
                    level3a = level3a + [val]
                    val = ' - '.join([str(to_percentage(v, percentage)) for v in scores[f'{category}_{avg}_{m}']['95% CI']])
                    level3b = level3b + [f'[{val}]']

    for y_label in y_labels:
        for m in metrics:
            level1 = level1 + [y_label]        
            level2 = level2 + [m]
            val = to_percentage(scores[f'{y_label}_{m}']['mean'], percentage)
            level3a = level3a + [val]
            val = ' - '.join([str(to_percentage(v, percentage)) for v in scores[f'{y_label}_{m}']['95% CI']])
            level3b = level3b + [f'[{val}]']
    
    df = pd.DataFrame({'Category': level1, 'Metrics': level2, 'Mean': level3a, '95% CI': level3b}) 
    return df

def plot_regression_results_with_metrics(model=None, test_dataloader=None, y_true=None, y_pred=None):
    device = DEVICE

    if model is not None: 
        model.eval()  # Set the model to evaluation mode
        y_true = []
        y_pred = []
    
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                outputs = outputs.squeeze()  # Ensure the outputs are in the correct shape
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        # Convert lists to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

    # Calculate R2 score
    r2_metric = R2Score().to(device)
    r2_metric.update(torch.tensor(y_pred), torch.tensor(y_true))
    r2_value = r2_metric.compute().item()

    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)

    # Calculate additional metrics: MAE and RMSE
    mae_metric = MeanAbsoluteError().to(device)
    mse_metric = MeanSquaredError(squared=False).to(device)  # RMSE

    mae_metric.update(torch.tensor(y_pred), torch.tensor(y_true))
    mse_metric.update(torch.tensor(y_pred), torch.tensor(y_true))

    mae_value = mae_metric.compute().item()
    rmse_value = mse_metric.compute().item()

    # Linear regression for the regression line
    slope, intercept, _, _, _ = linregress(y_true, y_pred)
    regression_line = slope * y_true + intercept

    # Plotting the true values vs predicted values
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, label="Predictions", alpha=0.7)
    plt.plot(y_true, regression_line, 'b-', lw=2, label="Regression Line")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label="Ideal fit (1:1)")

    # Add labels, title, and metrics
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(
        f"Regression Results: True vs. Predicted\n"
        f"R2: {r2_value:.4f}, Spearman: {spearman_corr:.4f}\n"
        f"MAE: {mae_value:.4f}, RMSE: {rmse_value:.4f}"
    )
    plt.legend()
    plt.grid(True)
    plt.show()

def get_centers_pred_labels(center: str, model: str, task:str):
    csv_path = f'/media/data1/achilsowa/results/fairseq/centers/{center}/{model}_{task}.csv'
    df = pd.read_csv(csv_path)

    if task == 'labels_77':
        y_labels =  ['Acute pericarditis', 'QS complex in V1-V2-V3', 'T wave inversion (anterior - V3-V4)', 'Right atrial enlargement','2nd degree AV block - mobitz 1','Left posterior fascicular block','Wolff-Parkinson-White (Pre-excitation syndrome)','Junctional rhythm','Premature ventricular complex',"rSR' in V1-V2",'Right superior axis','ST elevation (inferior - II, III, aVF)','Afib','ST elevation (anterior - V3-V4)','RV1 + SV6 > 11 mm','Sinusal','Monomorph','Delta wave','R/S ratio in V1-V2 >1','Third Degree AV Block','LV pacing','Nonspecific intraventricular conduction delay','ST depression (inferior - II, III, aVF)','Regular','Premature atrial complex','2nd degree AV block - mobitz 2','Left anterior fascicular block','Q wave (septal- V1-V2)','Prolonged QT','Left axis deviation','Left ventricular hypertrophy','ST depression (septal- V1-V2)','Supraventricular tachycardia','Atrial paced','Q wave (inferior - II, III, aVF)','no_qrs','T wave inversion (lateral -I, aVL, V5-V6)','Right bundle branch block','ST elevation (septal - V1-V2)','SV1 + RV5 or RV6 > 35 mm','Right axis deviation','RaVL > 11 mm','Polymorph','Ventricular tachycardia','QRS complex negative in III','ST depression (lateral - I, avL, V5-V6)','1st degree AV block','Lead misplacement','Q wave (posterior - V7-V9)','Atrial flutter','Ventricular paced','ST elevation (posterior - V7-V8-V9)','Ectopic atrial rhythm (< 100 BPM)','Early repolarization','Ventricular Rhythm','Irregularly irregular','Atrial tachycardia (>= 100 BPM)','R complex in V5-V6','ST elevation (lateral - I, aVL, V5-V6)','Brugada','Bi-atrial enlargement','Q wave (lateral- I, aVL, V5-V6)','ST upslopping','T wave inversion (inferior - II, III, aVF)','Regularly irregular','Bradycardia','qRS in V5-V6-I, aVL','Q wave (anterior - V3-V4)','Acute MI','ST depression (anterior - V3-V4)','Right ventricular hypertrophy','T wave inversion (septal- V1-V2)','ST downslopping','Left bundle branch block','Low voltage','U wave','Left atrial enlargement']
        bert = pd.read_parquet(f'/media/data1/achilsowa/results/fairseq/centers/bert_threshold.parquet')
        bert.index = bert['labels']
        thresholds = bert.loc[y_labels, ['threshold']].to_numpy().T
        b_labels = [f'{y}_bert_model' for y in y_labels]
        s_labels = [f'{y}_sig_model' for y in y_labels]
        prob_y_true = df.loc[:, b_labels].to_numpy()
        y_pred = df.loc[:, s_labels].to_numpy()
        y_true = prob_y_true - thresholds
        y_true = np.where(prob_y_true < thresholds, 0, 1)
    else:
        y_true = df.loc[:, ['ground_truth']].to_numpy()
        y_pred = df.loc[:, ['predictions']].to_numpy()
        
    return y_pred, y_true

def get_pred_labels(
    header_pkl,
    header_npy,
    DatasetClass=None,
    manifest_path=None,
    df_path=None,
    npy_path=None,
    y_idx=None,
    y_labels=None,
    sample_rate=250
):
    csv_path=parquet_path=None
    if df_path is not None:
        ext = df_path.split('.')[-1]
        if ext == 'csv':
            csv_path=df_path
        elif ext == 'parquet':
            parquet_path=df_path
    if header_pkl is not None:
        header = np.load(header_pkl, allow_pickle=True)
        y_pred = np.memmap(header_npy, 
            mode='r',
            shape=header['shape'],
            dtype=header['dtype']
        )
    else:
        y_pred = np.load(header_npy)


    if manifest_path is not None:
        dataset = DatasetClass(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            label=True
        )
        y_true = np.array([d['label'] for d in dataset])
    elif parquet_path is not None or csv_path is not None:
        df = pd.read_parquet(parquet_path) if csv_path is None else pd.read_csv(csv_path)
        if 'RV1 + SV6\xa0> 11 mm' in df.columns.tolist():
            df.rename(columns={'RV1 + SV6\xa0> 11 mm': 'RV1 + SV6 > 11 mm'}, inplace=True)
        df.reset_index(inplace=True, drop=True)
        y_true = df[y_labels].to_numpy()
    elif npy_path is not None:
        y_true = np.load(npy_path)
        y_true = y_true[:, y_idx]
    return y_pred, y_true

def get_path(key, root_dir='nas'):
    root_dir = ROOT_DIR_NAS if root_dir == 'nas' else ROOT_DIR_VOL
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get the depth of the current directory relative to the root
        depth = len(os.path.relpath(dirpath, root_dir).split(os.sep))
        # Only proceed if the depth is less than or equal to 2 (the first two levels)
        if depth <= 2:
            for dirname in dirnames:
                if dirname.endswith(key):
                    matching_dirs.append(os.path.join(dirpath, dirname))
        else: 
            # Stop walking deeper if we've reached beyond the second level
            del dirnames[:]  # Prevents os.walk from going deeper
    assert len(matching_dirs) == 1, f'key = {key}. Expected one directory but found {matching_dirs}'
    return matching_dirs[0]

def read_manifest(tsv_folder, subset,):
    header_pkl=f"{tsv_folder}/outputs_{subset.split('/')[-1]}_header.pkl"
    header_npy=f"{tsv_folder}/outputs_{subset.split('/')[-1]}.npy"
    
    df_path = npy_true_path = y_idx = y_labels = None
    tsv_path = os.path.join(tsv_folder, f'{subset}.tsv')
    with open(tsv_path,"r") as f:
        for i, line in enumerate(f):
            items = line.strip().split(":")
            assert len(items) == 2, line
            key, value = items
            if key == "y_path":
                npy_true_path = value
            if key == "y_pred":
                header_npy = value
                header_pkl = None
            elif key == "label_indexes":
                y_idx = eval(value)
            elif key == "y_labels":
                y_labels = eval(value)
            elif key == "x_path":
                df_path = value
    return header_npy, header_pkl, df_path, npy_true_path, y_idx, y_labels

class AbstractModel:
    def __init__(self, key: str, ds_folder: str, results_folder: str):
        self.key = key
        self.ds_folder = ds_folder
        self.results_folder = results_folder
    def get_results_folder(self,  task: AbstractTask) -> str:
        pass
    def get_ds_folder(self,  task: AbstractTask) -> str:
        pass
        
class DeepECGSL(AbstractModel):
    def __init__(self, key='sl', ds_folder='/media/data1/anolin/for_achille_ssl/for_table', results_folder='/media/data1/anolin/for_achille_ssl/for_table'):
        super().__init__(key, ds_folder, results_folder)
    def get_results_folder(self, task: AbstractTask):
        return os.path.join(self.results_folder, task.key)
    def get_ds_folder(self, task: AbstractTask):
        return os.path.join(self.ds_folder, task.key)
    
class DeepECGSSL(AbstractModel):
    def __init__(self, key='ssl', ds_folder='/media/data1/achilsowa/datasets/fairseq/mhi-mimic-code15/manifest/finetune'):
        super().__init__(key, ds_folder, '')
    def get_results_folder(self, task: AbstractTask):
        return get_path(task.results_key)
    def get_ds_folder(self, task: AbstractTask):
        return get_path(task.results_key)
    
class AbstractTask:
    def __init__(self, key, ds:AbstractDataset, model: AbstractModel, results_key, type='classification'):        
        self.key = key
        self.type = type
        self.metrics = ['auroc', 'f1score', 'auprc'] if type == 'classification' else ['mse', 'mae', 'r2', 'pearson_correlation']
        self.labels = []
        self.categories = {}
        self.ds = ds
        self.model = model
        self.results_key = results_key
        self.results_folder = ds.get_results_folder(self)

    def get_scores(self, force_evaluation=False):
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
                num_permutations=N_PERM,
                sample_ratio=RATIO
            )    
        else:
            scores = calculate_permuted_regression_metrics( 
                y_true=y_true, 
                y_pred=y_pred, 
                device=DEVICE, 
                y_labels=self.labels, 
                categories=self.categories, 
                num_permutations=N_PERM,
                sample_ratio=RATIO
            )

        fscores = format_ci_metrics(scores, self.categories, self.labels, self.metrics)
        fscores.to_csv(results_path)
        return fscores
    
        
class Labels77Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('labels_77', ds, model, 'last-ft-labels-77-bce')
        if ds.key == 'ssl':
            self.labels =  ['Acute pericarditis', 'QS complex in V1-V2-V3', 'T wave inversion (anterior - V3-V4)', 'Right atrial enlargement','2nd degree AV block - mobitz 1','Left posterior fascicular block','Wolff-Parkinson-White (Pre-excitation syndrome)','Junctional rhythm','Premature ventricular complex',"rSR' in V1-V2",'Right superior axis','ST elevation (inferior - II, III, aVF)','Afib','ST elevation (anterior - V3-V4)','RV1 + SV6 > 11 mm','Sinusal','Monomorph','Delta wave','R/S ratio in V1-V2 >1','Third Degree AV Block','LV pacing','Nonspecific intraventricular conduction delay','ST depression (inferior - II, III, aVF)','Regular','Premature atrial complex','2nd degree AV block - mobitz 2','Left anterior fascicular block','Q wave (septal- V1-V2)','Prolonged QT','Left axis deviation','Left ventricular hypertrophy','ST depression (septal- V1-V2)','Supraventricular tachycardia','Atrial paced','Q wave (inferior - II, III, aVF)','no_qrs','T wave inversion (lateral -I, aVL, V5-V6)','Right bundle branch block','ST elevation (septal - V1-V2)','SV1 + RV5 or RV6 > 35 mm','Right axis deviation','RaVL > 11 mm','Polymorph','Ventricular tachycardia','QRS complex negative in III','ST depression (lateral - I, avL, V5-V6)','1st degree AV block','Lead misplacement','Q wave (posterior - V7-V9)','Atrial flutter','Ventricular paced','ST elevation (posterior - V7-V8-V9)','Ectopic atrial rhythm (< 100 BPM)','Early repolarization','Ventricular Rhythm','Irregularly irregular','Atrial tachycardia (>= 100 BPM)','R complex in V5-V6','ST elevation (lateral - I, aVL, V5-V6)','Brugada','Bi-atrial enlargement','Q wave (lateral- I, aVL, V5-V6)','ST upslopping','T wave inversion (inferior - II, III, aVF)','Regularly irregular','Bradycardia','qRS in V5-V6-I, aVL','Q wave (anterior - V3-V4)','Acute MI','ST depression (anterior - V3-V4)','Right ventricular hypertrophy','T wave inversion (septal- V1-V2)','ST downslopping','Left bundle branch block','Low voltage','U wave','Left atrial enlargement']
        else:
            self.labels = ['Sinusal','Regular','Monomorph','QS complex in V1-V2-V3','R complex in V5-V6','T wave inversion (inferior - II, III, aVF)','Left bundle branch block','RaVL > 11 mm','SV1 + RV5 or RV6 > 35 mm','T wave inversion (lateral -I, aVL, V5-V6)','T wave inversion (anterior - V3-V4)','Left axis deviation','Left ventricular hypertrophy','Bradycardia','Q wave (inferior - II, III, aVF)','Afib','Irregularly irregular','Atrial tachycardia (>= 100 BPM)','Nonspecific intraventricular conduction delay','Premature ventricular complex','Polymorph','T wave inversion (septal- V1-V2)','Right bundle branch block','Ventricular paced','ST elevation (anterior - V3-V4)','ST elevation (septal - V1-V2)','1st degree AV block','Premature atrial complex','Atrial flutter',"rSR' in V1-V2",'qRS in V5-V6-I, aVL','Left anterior fascicular block','Right axis deviation','2nd degree AV block - mobitz 1','ST depression (inferior - II, III, aVF)','Acute pericarditis','ST elevation (inferior - II, III, aVF)','Low voltage','Regularly irregular','Junctional rhythm','Left atrial enlargement','ST elevation (lateral - I, aVL, V5-V6)','Atrial paced','Right ventricular hypertrophy','Delta wave','Wolff-Parkinson-White (Pre-excitation syndrome)','Prolonged QT','ST depression (anterior - V3-V4)','QRS complex negative in III','Q wave (lateral- I, aVL, V5-V6)','Supraventricular tachycardia','ST downslopping','ST depression (lateral - I, avL, V5-V6)','2nd degree AV block - mobitz 2','U wave','R/S ratio in V1-V2 >1','RV1 + SV6 > 11 mm','Left posterior fascicular block','Right atrial enlargement','ST depression (septal- V1-V2)','Q wave (septal- V1-V2)','Q wave (anterior - V3-V4)','ST upslopping','Right superior axis','Ventricular tachycardia','ST elevation (posterior - V7-V8-V9)','Ectopic atrial rhythm (< 100 BPM)','Lead misplacement','Third Degree AV Block','Acute MI','Early repolarization','Q wave (posterior - V7-V9)','Bi-atrial enlargement','LV pacing','Brugada','Ventricular Rhythm','no_qrs']
    
        self.categories = {
            "RHYTHM": ['Ventricular tachycardia','Bradycardia','Brugada','Wolff-Parkinson-White (Pre-excitation syndrome)','Atrial flutter','Ectopic atrial rhythm (< 100 BPM)','Atrial tachycardia (>= 100 BPM)','Sinusal','Ventricular Rhythm','Supraventricular tachycardia','Junctional rhythm','Regular','Regularly irregular','Irregularly irregular','Afib','Premature ventricular complex','Premature atrial complex'],
            "CONDUCTION": ['Left anterior fascicular block','Delta wave','2nd degree AV block - mobitz 2','Left bundle branch block','Right bundle branch block','Left axis deviation','Atrial paced','Right axis deviation','Left posterior fascicular block','1st degree AV block','Right superior axis','Nonspecific intraventricular conduction delay','Third Degree AV Block','2nd degree AV block - mobitz 1','Prolonged QT','U wave','LV pacing','Ventricular paced'],
            "CHAMBER ENLARGEMENT": ['Bi-atrial enlargement','Left atrial enlargement','Right atrial enlargement','Left ventricular hypertrophy','Right ventricular hypertrophy'],
            "PERICARDITIS": ['Acute pericarditis'],
            'INFARCT, ISCHEMIA': ['Q wave (septal- V1-V2)','ST elevation (anterior - V3-V4)','Q wave (posterior - V7-V9)','Q wave (inferior - II, III, aVF)','Q wave (anterior - V3-V4)','ST elevation (lateral - I, aVL, V5-V6)','Q wave (lateral- I, aVL, V5-V6)','ST depression (lateral - I, avL, V5-V6)','Acute MI','ST elevation (septal - V1-V2)','ST elevation (inferior - II, III, aVF)','ST elevation (posterior - V7-V8-V9)','ST depression (inferior - II, III, aVF)','ST depression (anterior - V3-V4)'],
            "OTHER": ['ST downslopping','ST depression (septal- V1-V2)','R/S ratio in V1-V2 >1','RV1 + SV6 > 11 mm','Polymorph',"rSR' in V1-V2",'QRS complex negative in III','qRS in V5-V6-I, aVL','QS complex in V1-V2-V3','R complex in V5-V6','RaVL > 11 mm','T wave inversion (septal- V1-V2)','SV1 + RV5 or RV6 > 35 mm','T wave inversion (inferior - II, III, aVF)','Monomorph','T wave inversion (anterior - V3-V4)','T wave inversion (lateral -I, aVL, V5-V6)','Low voltage','Lead misplacement','ST depression (anterior - V3-V4)','Early repolarization','ST upslopping','no_qrs'],
            "ALL": self.labels
        }   
        
class Afib5Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('afib_5', ds, model, 'last-ft-afib-v2-5-bf')
        self.labels = ['label_5y']

class Lvef40Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('lvef_40', ds, model, 'last-ft-fevg-v2-40-bce')
        self.labels = ['LVEF_EQUAL_OR_UNDER_40_tte_lvef']

class Lvef50Task(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('lvef_50', ds, model, 'last-ft-fveg-40-bce')
        self.labels =   ['LVEF_UNDER_50_tte_lvef']

class LvefRegTask(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('lvef_reg', ds, model, 'last-ft-fevg-reg-mse', type='regression')
        ['Visually Estimated EF_tte_lvef']

class LqtsTask(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('lqts', ds, model, 'last-ft-lqts-bf')
        self.labels = ['LQTS']

class LqtsTypeTask(AbstractTask):
    def __init__(self, ds:AbstractDataset, model: AbstractModel):
        super().__init__('lqts_type', ds, model, 'last-ft-lqts-type-bf')
        self.labels = ['LQTS_TYPE_1']

class AbstractDataset:
    def __init__(self, key: str, is_center: bool, tasks: List[str]):
        self.tasks = tasks
        self.is_center = is_center
        self.key = key
        self.results_folder=''

    def get_results_folder(self, task: AbstractTask):
        pass
    def get_results_path(self, task: AbstractTask):
        pass
    def get_y(self, task: AbstractTask):
        pass
    def get_stats(self, task: AbstractTask, force_evaluation=False):
        ds_folder = task.model.get_ds_folder(task)
        stats_path = os.path.join(ds_folder, f'stats_{task.model.key}.txt')
        if not force_evaluation and os.path.exists(stats_path):
            with open(stats_path, 'rb') as file:
                return pickle.load(file)
            
            _, Y = self.get_y(task)
            stats = {'$count$': Y.shape[0]}
            Y_n = np.sum(Y, axis=0)
            for i, label in enumerate(task.labels):
                stats[label] = Y_n[i]
            for key, cat in task.categories.items():
                idx = [self.labels.index(i) for i in cat]
                Y_n = Y[:, idx]
                stats[key] = np.sum(np.max(Y_n, axis=1), axis=0)
            with open(stats_path, 'wb') as file:
                pickle.dump(stats, file)
            return stats

class PublicDataset(AbstractDataset):
    def __init__(self, key, tasks):
        super().__init__(key, False, tasks)
    def get_results_folder(self, task):
        return task.model.get_results_folder(task)
    def get_results_path(self, task: AbstractTask):
        results_folder = task.model.get_results_folder(task)
        return os.path.join(results_folder, ci_scores_filename(self.key))    
    def get_y(self, task: AbstractTask):
        header_npy, header_pkl, df_path, npy_path, y_idx, y_labels = read_manifest(task.model.get_ds_folder(task), task.ds.key)
        return get_pred_labels(
            header_pkl=header_pkl,
            header_npy=header_npy,
            df_path=df_path,
            npy_path=npy_path,
            y_idx=y_idx,
            y_labels=y_labels
        )
    
class CenterDataset(AbstractDataset):
    def __init__(self, key, tasks):
        super().__init__(key, True, tasks)
        self.results_folder = f'/media/data1/achilsowa/results/fairseq/centers/{key}'
    def get_y(self, task: AbstractTask):
        return get_centers_pred_labels(self.key, task.model.key, task.key) 
    def get_results_path(self, task:AbstractTask):
        return os.path.join(self.results_folder, ci_scores_filename(f'{task.model.key}-{task.key}'))        
    def get_results_folder(self, task):
        return self.results_folder
        
def getSingletonModel(type: str) -> AbstractModel:
    assert type in MODELS
    key = f'model_{type}'
    if key in SINGLETON: 
        return SINGLETON[key]
    elif type == 'sl':
        model = SINGLETON[key] = DeepECGSL()
        return model
    elif type == 'ssl':
        model = SINGLETON[key] = DeepECGSSL()
        return model
    
def getSingletonDataset(subset: str) -> AbstractDataset:
    assert subset in DATASETS
    key = f'ds_{subset}'
    if key in SINGLETON:
        return SINGLETON[key]
    elif subset == ['mhi', 'test']:
        ds = SINGLETON[key] = PublicDataset('mhi', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50', 'lqts', 'lqts_type'])
        return ds
    elif subset == 'mimic':
        ds = SINGLETON[key] = PublicDataset('mimic_cleaned', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset in ['ptb', 'ptb_cleaned']:
        ds = SINGLETON[key] = PublicDataset('ptb_cleaned', tasks=['labels_77'])
        return ds
    elif subset in ['clsa', 'clsa_cleaned']:
        ds = SINGLETON[key] = PublicDataset('clsa_cleaned', tasks=['labels_77'])
        return ds
    elif subset in ['ukb', 'ukbb_cleaned_high_pass_scaled']:
        ds = SINGLETON[key] = PublicDataset('ukbb_cleaned_high_pass_scaled', tasks=['labels_77'])
        return ds
    elif subset == 'external_public':
        ds = SINGLETON[key] = PublicDataset('external_public', tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'uw':
        ds = SINGLETON[key] = CenterDataset(subset, tasks=['labels_77', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'ucsf':
        ds = SINGLETON[key] = CenterDataset(subset, tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'cshs':
        ds = SINGLETON[key] = CenterDataset(subset, tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'hms':
        ds = SINGLETON[key] = CenterDataset(subset, tasks=['labels_77'])
        return ds
    elif subset == 'nyp':
        ds = SINGLETON[key] = CenterDataset(subset, tasks=['labels_77', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'jgh':
        ds = SINGLETON[key] = CenterDataset(subset, tasks=['labels_77', 'lvef_40', 'lvef_50'])
        return ds
    elif subset == 'external_private':
        ds = SINGLETON[key] = CenterDataset(subset, tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50'])
        return ds

def getSingletonTask(task_key:str, model_key:str, ds_key: str) -> AbstractTask:
    assert task_key in TASKS
    model = getSingletonModel(model_key)
    ds = getSingletonDataset(ds_key)
    key = f'task_{task_key}'
    if key in SINGLETON:
        return SINGLETON[key]
    elif task_key == 'labels_77':
        task = SINGLETON[key] = Labels77Task(ds, model)
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



if __name__ == '__main__':
    sl = getSingletonModel('sl')
    ssl = getSingletonModel('ssl')
    models = [sl, ssl]
    
    def evaluate_datasets(datasets: List[str]):
        for ds in datasets:
            dataset = getSingletonDataset(ds)
            for key in dataset.tasks:
                for model in [ssl]: #, ssl]:
                    task = getSingletonTask(key, model.key, dataset.key)
                    print('='*10, f'evaluation of task: dataset: {dataset.key}, {task.key}, model: {model.key}', '='*10)
                    task.get_scores(force_evaluation=True)

    #def evaluate_stats(datasets = TASKS):
    #    for ds in datasets:
    #        for 
    centers = ['uw', 'ucsf', 'jgh', 'nyp', 'hms', 'cshs', 'external_private']
    evaluate_datasets(['external_private'])