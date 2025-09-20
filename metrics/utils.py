from __future__ import annotations  # Enables forward references
from typing import Any, List

import pandas as pd
import numpy as np
import torch
import os
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, mean_absolute_error, auc
import torchmetrics.functional.classification as Fc

from scipy.stats import spearmanr, linregress, pearsonr, norm
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm 
import pickle


ROOT_DIR_NAS='/media/data1/achilsowa/results/fairseq/outputs/'
ROOT_DIR_VOL='/volume/deepecg/fairseq-signals/outputs/'
ROOT_DIR = 'nas'
EVAL_CI = FORMAT = False
DEVICE = "cuda:1"
N_PERM = 1000
RATIO = 0.7

TASKS = ['labels_77', 'afib_5', 'lvef_40', 'lvef_50', 'lqts', 'lqts_type', 'labels_47', 'labels_14']
DATASETS = ['mhi', 'mimic', 'ptb', 'ukb', 'clsa', 'external_public', 'uw', 'ucsf', 'jgh', 'nyp', 'hms', 'cshs', 'chum', 'external_private', 'external', 'train']
MODELS = ['sl', 'e2e-sl', 'ssl', 'e2e-ssl', 'ecg-founder', 'ecg-fm']
SINGLETON = {}


def set_DEVICE(device):
    DEVICE = device

def set_N_PERM(n_perm):
    N_PERM = n_perm

def set_RATIO(ratio):
    RATIO = ratio


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

def to_percentage(num, percentage=True):
    return f'{float(num)*100:.2f}' if percentage else f'{float(num):.3f}'
    
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
    t_y_pred_threshold_youden = (
        torch.tensor(y_pred_thresholded_youden, device=device, dtype=torch.long)
        if youden_threshold
        else None
    )
    scores = {"tm": {}}
    task = "binary"
    
    if single:
        for id, y_label in enumerate(y_labels):
            t_pred = t_y_pred[:, id]
            t_target = t_y_true[:, id]
            # t_pred_threshold_default = t_y_pred_threshold_default[:, id]

            scores["tm"][f"{y_label}_auroc"] = try_catch_fn(lambda: Fc.auroc(t_pred, t_target, task=task))
            scores["tm"][f"{y_label}_auprc"] = try_catch_fn(lambda: Fc.average_precision(t_pred, t_target, task=task))
            scores["tm"][f"{y_label}_f1score"] = try_catch_fn(lambda: Fc.f1_score(t_pred, t_target, task=task))
            scores["tm"][f"{y_label}_acc"] = try_catch_fn(lambda: Fc.accuracy(t_pred, t_target, task=task))
            scores["tm"][f"{y_label}_bacc"] = try_catch_fn(lambda: Fc.accuracy(t_y_pred_threshold[:, id], t_target, task='multiclass', num_classes=2, average='macro'))
            # scores["tm"][f"{y_label}_micro_ppv_default"] = try_catch_fn(lambda: Fc.precision(t_pred_threshold_default, t_target, task=task, average="micro"))
            # PPV with Youden's threshold (if applicable)
            # if youden_threshold:
                # t_pred_threshold_youden = t_y_pred_threshold_youden[:, id]
                # scores["tm"][f"{y_label}_micro_ppv_youden"] = try_catch_fn(lambda: Fc.precision(t_pred_threshold_youden, t_target, task=task, average="micro"))
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


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    #assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    #order = (~ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

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
    y_pred_idx=None,
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
    
    if y_pred_idx is not None:
        y_pred = y_pred[:, y_pred_idx]


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
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)

        y_true = y_true[:, y_idx]
    
    if np.any(y_pred < 0):
        y_pred = stable_sigmoid(y_pred)
    return y_pred, y_true

def get_scores(y_pred, y_true, y_labels, categories,  task='classification'):
    if task == 'classification':
        scores = calculate_permuted_metrics(
            y_true=y_true, 
            y_pred=y_pred, 
            device=DEVICE, 
            y_labels=y_labels, 
            categories=categories, 
            num_permutations=N_PERM,
            sample_ratio=RATIO
        )    
    else:
        scores = calculate_permuted_regression_metrics(
            y_true=y_true, 
            y_pred=y_pred, 
            device=DEVICE, 
            y_labels=y_labels, 
            categories=categories, 
            num_permutations=N_PERM,
            sample_ratio=RATIO
        )
    return scores

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

def read_manifest(tsv_folder: str, results_folder: str, subset: str):
    y_pred_pkl=f"{results_folder}/outputs_{subset.split('/')[-1]}_header.pkl"
    y_pred_path=f"{results_folder}/outputs_{subset.split('/')[-1]}.npy"
    
    df_path = y_path = y_idx = y_pred_idx = y_labels = None
    tsv_path = os.path.join(tsv_folder, f'{subset}.tsv')
    with open(tsv_path,"r") as f:
        for i, line in enumerate(f):
            items = line.strip().split(":")
            assert len(items) == 2, line
            key, value = items
            if key == "y_path":
                y_path = value
            if key == "y_pred":
                y_pred_pkl = None
                y_pred_path = value
            elif key == "label_indexes":
                y_idx = eval(value)
            elif key == "label_pred_indexes":
                y_pred_idx = eval(value)
            elif key == "y_labels":
                y_labels = eval(value)
            elif key == "x_path":
                df_path = value
    return y_pred_pkl, y_pred_path, y_pred_idx, df_path, y_path, y_idx, y_labels



def merge_centers_scores(tasks=['labels_77', 'afib_5', 'lvef_40', 'lvef_50', 'lqts', 'lqts_type']):
    def merge_fn(centers):
        for model in ['sl', 'ssl']:
            dfs = []
            for center in centers:
                df = df = pd.read_csv(f'/media/data1/achilsowa/results/fairseq/centers/{center}/{model}_{task}.csv')
                df['center'] = center
                dfs += [df]
            dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
            dfs.to_csv(f'/media/data1/achilsowa/results/fairseq/centers/external_private/{model}_{task}.csv')

    for task in tasks:
        if task == 'labels_77':
            merge_fn(['uw', 'ucsf', 'hms', 'cshs', 'jgh', 'nyp'])
        if task == 'afib_5':
            merge_fn(['ucsf', 'cshs'])
        elif task == 'lvef_40':
            merge_fn(['uw', 'ucsf', 'cshs', 'jgh', 'nyp'])
        elif task == 'lvef_50':
            merge_fn(['uw', 'ucsf', 'cshs', 'jgh', 'nyp'])

def merge_public_ds_scores():
    def merge_sl_fn():
        for label in ['ground', 'pred']:
            subsets = ['MIMIC', 'PTB', 'CLSA', 'UKB']
            Y = [np.load(f'/media/data1/anolin/for_achille_ssl/for_table/{subset}_LABELS_77/Y_{label}.npy') for subset in subsets]
            Y = np.concatenate(Y, axis=0)
            np.save(f'/media/data1/anolin/for_achille_ssl/for_table/EXTERNAL_PUBLIC_LABELS_77/Y_{label}.npy', Y)
        
    def merge_ssl_fn():
        result_path = get_path('last-ft-labels-77-bce')
        def save_npy(y):
            header = {'shape': y.shape, 'dtype': y.dtype}
            with open(f'{result_path}/outputs_external_public_header.pkl', 'wb') as f:
                pickle.dump(header, f)
            #with open(f'{result_path}/outputs_external_public.npy', 'wb') as f:
            f = f'{result_path}/outputs_external_public.npy'
            memmap = np.memmap(f, dtype=y.dtype, mode='w+', shape=y.shape)
            memmap[:] = y[:]  # Copy data to the memory-mapped file
            del memmap  # Ensure changes are written to disk
        
        y_trues = []
        y_preds = []
        subsets = ['mimic_cleaned', 'ptb_cleaned', 'clsa_cleaned', 'ukbb_cleaned_high_pass_scaled']
        subsets = ['clsa_cleaned']
        for subset in subsets:            
            header_npy, header_pkl, df_path, npy_path, y_idx = read_manifest('last-ft-labels-77-bce', 'nas', f'labels-77/{subset}')
            y_pred, y_true = get_pred_labels(
                header_pkl=header_pkl,
                header_npy=header_npy,
                df_path=df_path,
                npy_path=npy_path,
                y_idx=list(range(77)),
                y_labels=None
            )
            y_trues += [y_true]
            y_preds += [y_pred]
        
        Y = np.concatenate(y_trues, axis=0)
        np.save('/media/data1/anolin/for_achille_ssl/NEW_DS/EXTERNAL_PUBLIC/Y.npy', Y)

        Y = np.concatenate(y_preds, axis=0)
        save_npy(Y)

    merge_sl_fn()
    merge_ssl_fn()




def plot_histo_old(models, categories, title, filename, metric='auroc'):
    # Sample data
    # Sample data
    def get_fig_width():
        len_cat = len(categories)
        return len_cat + 2
    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.15  # Width of the bars

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(get_fig_width(), 6))

    # Plot bars without error caps using seaborn for improved aesthetics
    # The radar chart needs the data to be a closed loop, so append the start to the end.
    for model in models:          
        scores = model['df'].loc[list(categories.keys()), 'Mean'].tolist()
        cis = model['df'].loc[list(categories.keys()), '95% CI'].tolist()
        model['scores'] = scores
        model['cis'] = [eval(ci.replace('-', ',')) for ci in cis]
    # Plot each model’s data on the radar chart
    m1, m2 = models[0], models[1]
    ax.bar(x - width/2, m1['scores'], width, label=m1['title'], color=m1['color'])
    ax.bar(x + width/2, m2['scores'], width, label=m2['title'], color=m2['color'])

    # Add thin vertical lines for CI
    for i in range(len(categories)):
        ax.vlines(x[i] - width/2, m1['cis'][i][0], m1['cis'][i][1], color='gray', linewidth=1)
        ax.vlines(x[i] + width/2, m2['cis'][i][0], m2['cis'][i][1], color='gray', linewidth=1)
        
        # Calculate and add delta (difference) text
        delta = round(m2['scores'][i] - m1['scores'][i], 2)
        y_position = max(m1['cis'][i][1], m2['cis'][i][1]) + 0.5  # Position above bars
        ax.text(x[i], y_position, f'Δ={delta}', ha='center', va='bottom', fontsize=10, color='black')

    # Labeling and aesthetics
    #ax.set_xlabel('Categories')
    ax.set_ylabel(metric.upper(), fontsize=15)
    ax.set_title(title, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories.values(), rotation=15, ha='right', fontsize=15)
    ax.legend(loc='lower right')

    # Show plot
    plt.tight_layout()
    plt.savefig(f"/volume/deepecg/fairseq-signals/metrics/images/{filename}.png", format="png", dpi=300)  # Save as a high-quality PNG file
    
    plt.show()



def plot_histo(models, categories, title, filename, fontsize=16, y_start=.50, y_end=1.00, hide_last_y=False, show_p_value=False, metric='auroc'):
    # Sample data
    # Sample data
    def get_fig_width():
        len_cat = len(categories)
        return len_cat + 3
    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.35  # Width of the bars

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(get_fig_width(), 6))

    # Plot bars without error caps using seaborn for improved aesthetics
    # The radar chart needs the data to be a closed loop, so append the start to the end.
    for model in models:          
        scores = model['df'].loc[list(categories.keys()), 'Mean'].tolist()
        cis = model['df'].loc[list(categories.keys()), '95% CI'].tolist()
        model['scores'] = scores
        model['cis'] = [eval(ci.replace('-', ',')) for ci in cis]
        if show_p_value:
            if 'p_value' in model['df'].columns:
                model['p_values'] = model['df'].loc[list(categories.keys()), 'p_value'].tolist()

    # Plot each model’s data on the radar chart
    gap = 0.01
    m1, m2 = models[0], models[1]
    bars_m1 = ax.bar(x - width/2 - gap, m1['scores'], width, label=m1['title'], color=m1['color'])
    bars_m2 = ax.bar(x + width/2 + gap, m2['scores'], width, label=m2['title'], color=m2['color'])

    # Add thin vertical lines for CI
    for i in range(len(categories)):
        ax.vlines(x[i] - width/2, m1['cis'][i][0], m1['cis'][i][1], color='gray', linewidth=1)
        ax.vlines(x[i] + width/2, m2['cis'][i][0], m2['cis'][i][1], color='gray', linewidth=1)
        
        # Calculate and add delta (difference) text
        delta = round(m2['scores'][i] - m1['scores'][i], 2)
        y_position = max(m1['cis'][i][1], m2['cis'][i][1]) + 0.5  # Position above bars
        if show_p_value:
            p_val_text = "P < 0.001" if m2['p_values'][i] < 0.001 else f"P = {m2['p_values'][i]:.3f}"
            ax.hlines(y_position+0.5, i - width + gap*5 , i + width - gap*5, color='black', linewidth=1)
            ax.text(x[i], y_position+1., p_val_text, ha='center', va='bottom', fontsize=fontsize, color='black')
        else:
            ax.text(x[i], y_position, f'Δ={delta}', ha='center', va='bottom', fontsize=fontsize, color='black')

    # Add values inside the bars
    for bar in bars_m1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height - 10, f'{height:.2f}', ha='center', va='bottom', color='white', rotation=70, fontsize=fontsize - 6*0)

    for bar in bars_m2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height - 10, f'{height:.2f}', ha='center', va='bottom', color='white', rotation=70, fontsize=fontsize - 6*0)

    # Labeling and aesthetics
    #ax.set_xlabel('Categories')
    ax.set_ylabel(metric.upper(), fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(categories.values(), rotation=15, ha='right', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    #ax.legend(loc='upper left')

    # Show plot
    plt.ylim(y_start, y_end)
    if hide_last_y:
        ax.yaxis.get_major_ticks()[-1].draw = lambda *args:None
    # plt.tight_layout()
    # plt.savefig(f"/volume/deepecg/fairseq-signals/metrics/images/{filename}.png", format="png", dpi=300)  # Save as a high-quality PNG file
    
    plt.show()


def save_pred_in_df(df, ds, task, save_gt=False):
    y_pred, y_true = ds.get_y(task)
    for idx, label in enumerate(task.labels):
        df[f'{label}_pred'] = y_pred[:, idx] 
        if save_gt: 
            df[f'{label}_gt'] = y_true[:, idx] 
    return df


def plot_spider(models, categories, title, filename, y_start=.70, y_end=1.00, fontsize=16, num_ticks=5, fig_size=None):
    
    def get_fig_width():
        if fig_size is not None:
            return fig_size
        len_cat = len(categories)
        return len_cat + 2
    
    # Set up bar positions
    # Number of variables we're plotting
    num_vars = len(categories)

    # Compute angle of each category for the radar plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The radar chart needs the data to be a closed loop, so append the start to the end.
    for model in models:          
        scores = model['df'].loc[list(categories.keys()), 'Mean'].tolist()
        scores += scores[:1]
        model['scores'] = scores
    angles += angles[:1]

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(get_fig_width(), get_fig_width()), subplot_kw=dict(polar=True))

    # Plot each model’s data on the radar chart
    for model in models:
        ax.plot(angles, model['scores'], linewidth=1.5, linestyle='solid', label=model['title'], color=model['color'])
        ax.fill(angles, model['scores'], color=model['color'], alpha=0.3)
        ax.scatter(angles[:-1], model['scores'][:-1], color=model['color'], s=50, zorder=5)  # Exclude the repeated first point

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories.values(), ha='center')
    angles = np.array(angles)
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    angles = np.rad2deg(angles) 
    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        x,y = label.get_position()
        #print(x, y, angle, label.get_text())
        delta_y = .105
        if 340 < angle and angle < 380 and len(label.get_text()) > 5:
            delta_y = .25
        lab = ax.text(x,y-delta_y, label.get_text(), transform=label.get_transform(),
            ha=label.get_ha(), va=label.get_va(), fontsize=fontsize)
        #lab.set_rotation(angle) 
        labels.append(lab)
    #ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    #ax.set_xticklabels(categories.values(), ha='center')


    # Set the range for the radial axis (AUC percentage from 0 to 100)
    ax.set_ylim(ymin=y_start, ymax=y_end) #, num_ticks)
    ax.tick_params(axis='y', labelsize=fontsize)
    # Remove default x-ticks as we use custom labels
    #ax.set_xticks([])
        
    # Add title and legend
    ax.set_title(title, fontsize=fontsize, color='black', y=1.1)
    #ax.legend(loc='lower right', )
    plt.tight_layout()

        # Save the plot as a file
    plt.savefig(f"/volume/deepecg/fairseq-signals/metrics/images/{filename}.png", format="png", dpi=300)  # Save as a high-quality PNG file
    plt.show()


def sl_ssl_ds_size(tasks, title, start_y=70, fontsize=15, metric='auroc'):
    # Prepare data
    tasks_old = [
    {"name": "LQTS Type", "n_samples": 334, "sl_auroc": 85.04, "ssl_auroc": 93.06},
    {"name": "LQTS", "n_samples": 2741, "sl_auroc": 85.04, "ssl_auroc": 93.72},
    {"name": "LVEF <= 40", "n_samples": 89500, "sl_auroc": 93.72, "ssl_auroc": 93.57},
    {"name": "LVEF < 50", "n_samples": 89500, "sl_auroc": 98.53, "ssl_auroc": 93.75},
    {"name": "iAF5", "n_samples": 537742, "sl_auroc": 97.39, "ssl_auroc": 97.2},
    {"name": "ECG 77", "n_samples": 1166896, "sl_auroc": 93.17, "ssl_auroc": 93.96},
    ]
    #markers = [task["mark"] for task in tasks ] #"o", "s", "^", "^", "D", "x"]  # Circle, square, triangle, diamond
    markers = ["o", "s", "^", "^", "D", "x"]  # Circle, square, triangle, diamond
    n_samples = [task["n_samples"] for task in tasks]
    sl_auroc = [task["sl_auroc"] for task in tasks]
    ssl_auroc = [task["ssl_auroc"] for task in tasks]
    task_names = [task["name"] for task in tasks]
    colors = ['skyblue', 'salmon']
    colors = ['blue', 'green']

    # Plot SL and SSL points with unique markers
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_xlim(min(n_samples) * 0.8, max(n_samples) * 1.2)
    ax.set_ylim(start_y, 100)
    fontsize=15

    for i, task in enumerate(tasks):
        ax.plot(task["n_samples"], task["sl_auroc"], marker=markers[i], color=colors[0], label=f"SL ({task['name']})", alpha=0.8, linestyle="None")
        ax.plot(task["n_samples"], task["ssl_auroc"], marker=markers[i], color=colors[1], label=f"SSL ({task['name']})", alpha=0.8, linestyle="None")

    # Connect points for the same model with dotted lines
    ax.plot(n_samples, sl_auroc, linestyle="--", color=colors[0], alpha=0.6, label="SL (Trend)")
    ax.plot(n_samples, ssl_auroc, linestyle="--", color=colors[1], alpha=0.6, label="SSL (Trend)")

    # Add labels for each point
    for i, task in enumerate(tasks):
        ax.text(task["n_samples"], task["sl_auroc"], f"SL ({task['name']})", fontsize=fontsize, ha="left", color=colors[0])
        ax.text(task["n_samples"], task["ssl_auroc"], f"SSL ({task['name']})", fontsize=fontsize, ha="right", color=colors[1])


    # Plot SL and SSL points
    #ax.plot(n_samples, sl_auroc, "o--", color="blue", label="SL", alpha=0.8)
    #ax.plot(n_samples, ssl_auroc, "o--", color="green", label="SSL", alpha=0.8)

    # Add labels for each point
    #for i, (x, y) in enumerate(zip(n_samples, sl_auroc)):
    #    ax.text(x, y, f"SL ({task_names[i]})", fontsize=10, ha="right", color="blue")

    #for i, (x, y) in enumerate(zip(n_samples, ssl_auroc)):
    #    ax.text(x, y, f"SSL ({task_names[i]})", fontsize=10, ha="left", color="green")

    # Aesthetics
    ax.set_xlabel("Number of Samples (Log Scale)", fontsize=fontsize+2)
    ax.set_ylabel(metric.upper(), fontsize=fontsize+2)
    ax.set_title(title, fontsize=fontsize+4)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(visible=True, linestyle="--", alpha=0.6)

    # Show plot
    plt.tight_layout()
    plt.show()



def is_in(v, interval):
        return interval[0]<=v and v <= interval[1]
def delta(mean1, ci1, mean2, ci2):
    diff = float(mean2)-float(mean1)
    ci1_f = eval(ci1.replace(' - ', ',')) if isinstance(ci1, str) else ci1 
    ci2_f = eval(ci2.replace(' - ', ',')) if isinstance(ci2, str) else ci2 

    signif = -1 if diff < 0 else 1
    #diff = f'+{diff}' if diff > 0 else f'-{diff}'
    if is_in(ci1_f[0] , ci2_f) or is_in(ci1_f[1], ci2_f) or is_in(ci2_f[0], ci1_f) or is_in(ci2_f[1], ci1_f):
        signif = 0
    return diff, signif