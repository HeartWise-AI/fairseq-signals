import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from fairseq_signals import logging, metrics, meters
from fairseq_signals.data.ecg import ecg_utils
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

@dataclass
class BinaryFocalCriterionConfig(Dataclass):
    weight: Optional[List[float]] = field(
        default = None,
        metadata = {
            "help": "a manual rescaling weight given to the loss of each batch element."
            "if given, has to be a float list of size nbatch."
        }
    )
    weights_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "score weights file for cinc challenge, only used when --report_cinc_score is True"
        }
    )
    per_log_keys: List[str] = field(
        default_factory = lambda: [],
        metadata={
            "help": "additionally log metrics for each of these keys, only applied for acc, auc, f1score"
        }
    )
    alpha: float = field(
        default=0.25,
        metadata={"help": "Weighting factor for the rare class (usually the minority class), default is 0.25."}
    )
    gamma: float = field(
        default=2.0,
        metadata={"help": "Focusing parameter that adjusts the rate at which easy examples are down-weighted, default is 2.0."}
    )
    report_auc: bool = field(
        default=False,
        metadata={"help": "whether to report auprc / auroc metric, used for valid step"}
    )
    report_cinc_score: bool = field(
        default=False,
        metadata={"help": "whether to report cinc challenge metric"}
    )

@register_criterion(
    "binary_focal", dataclass = BinaryFocalCriterionConfig
)
class BinaryFocalCriterion(BaseCriterion):
    def __init__(self, cfg: BinaryFocalCriterionConfig, task: Task):
        super().__init__(task)
        self.weight = cfg.weight
        self.report_auc = cfg.report_auc
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma


    def forward(self, model, sample, reduce = True, save_outputs=False):
        """Compute the loss for the given sample.
        
        Returns a tuple with three elements.
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)
        if save_outputs:
            self.store(logits, target)

        probs = torch.sigmoid(logits)
        target = target.float()
        
        bce_loss = F.binary_cross_entropy(probs, target, reduction='none')


        # Calculate the modulating factor (1 - p_t)^gamma
        p_t = probs * target + (1 - probs) * (1 - target)
        modulating_factor = torch.pow(1 - p_t, self.gamma)

        # Apply the alpha factor
        alpha_factor = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Combine factors to compute the final focal loss
        loss = alpha_factor * modulating_factor * bce_loss

        # Apply reduction method
        if reduce:
            loss = loss.sum()
        
        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target.long().sum().item()
        
        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "nsignals": sample["id"].numel(),
            "sample_size": sample_size
        }
        
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            outputs = (probs > 0.5)

            if probs.numel() == 0:
                corr = 0
                count = 0
                tp = 0
                tn = 0
                fp = 0
                fn = 0
            else:
                count = float(probs.numel())
                corr = (outputs == target).sum().item()

                true = torch.where(target == 1)
                false = torch.where(target == 0)
                tp = outputs[true].sum()
                fn = outputs[true].numel() - tp
                fp = outputs[false].sum()
                tn = outputs[false].numel() - fp

            logging_output["correct"] = corr
            logging_output["count"] = count

            logging_output["tp"] = tp.item()
            logging_output["fp"] = fp.item()
            logging_output["tn"] = tn.item()
            logging_output["fn"] = fn.item()

            if not self.training and self.report_auc:
                logging_output["_y_true"] = target.cpu().numpy()
                logging_output["_y_score"] = probs.cpu().numpy()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        nsignals = utils.item(
            sum(log.get("nsignals", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", 0) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", 0) for log in logging_outputs])

            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true)

            if len(y_true) > 0:
                metrics.log_derived(
                    "auroc",
                    lambda meters: safe_round(
                        meters["_auc"].auroc, 3
                    )
                )
                metrics.log_derived(
                    "auprc",
                    lambda meters: safe_round(
                        meters["_auc"].auprc, 3
                    )
                )

        metrics.log_scalar("nsignals", nsignals)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        em_correct = sum(log.get("em_correct", 0) for log in logging_outputs)
        metrics.log_scalar("_em_correct", em_correct)

        em_total = sum(log.get("em_count", 0) for log in logging_outputs)
        metrics.log_scalar("_em_total", em_total)

        tp = sum(log.get("tp", 0) for log in logging_outputs)
        metrics.log_scalar("_tp", tp)
        fp = sum(log.get("fp", 0) for log in logging_outputs)
        metrics.log_scalar("_fp", fp)
        fn = sum(log.get("fn", 0) for log in logging_outputs)
        metrics.log_scalar("_fn", fn)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan")
            )

            metrics.log_derived(
                "em_accuracy",
                lambda meters: safe_round(
                    meters["_em_correct"].sum / meters["_em_total"].sum, 5
                )
                if meters["_em_total"].sum > 0
                else float("nan")
            )

            metrics.log_derived(
                "precision",
                lambda meters: safe_round(
                    meters["_tp"].sum / (meters["_tp"].sum + meters["_fp"].sum), 5
                )
                if (meters["_tp"].sum + meters["_fp"].sum) > 0
                else float("nan")
            )

            metrics.log_derived(
                "recall",
                lambda meters: safe_round(
                    meters["_tp"].sum / (meters["_tp"].sum + meters["_fn"].sum), 5
                )
                if (meters["_tp"].sum + meters["_fn"].sum) > 0
                else float("nan")
            )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsignals",
            "sample_size",
            "all_zeros",
            "all_zeros_t",
            "o_score",
            "c_score",
            "i_score",
            "correct",
            "count",
            "em_correct",
            "em_count",
            "tp",
            "fp",
            "tn",
            "fn",
            "_y_true",
            "_y_score"
        }

        for log_key in logging_outputs[0]:
            if log_key not in builtin_keys:
                if log_key.endswith("em_count"):
                    log_key = log_key.split("em_count")[0]
                    em_counts = [log[log_key + "em_count"] for log in logging_outputs]
                    em_corrects = [log[log_key + "em_correct"] for log in logging_outputs]
                    aggregated_em_counts = Counter()
                    aggregated_em_corrects = Counter()
                    for em_count, em_correct in zip(em_counts, em_corrects):
                        aggregated_em_counts.update(Counter(em_count))
                        aggregated_em_corrects.update(Counter(em_correct))
                    aggregated_em_counts = dict(aggregated_em_counts)
                    aggregated_em_corrects = dict(aggregated_em_corrects)

                    for log_id in aggregated_em_counts.keys():
                        key = log_key + str(log_id)

                        metrics.log_scalar(
                            "_" + key + "_em_total",
                            aggregated_em_counts[log_id]
                        )
                        metrics.log_scalar(
                            "_" + key + "_em_correct",
                            aggregated_em_corrects[log_id]
                        )

                        if aggregated_em_counts[log_id] > 0:
                            key1 = "_" + key + "_em_correct"
                            key2 = "_" + key + "_em_total"
                            metrics.log_derived(
                                key + "_em_accuracy",
                                lambda meters, key1=key1, key2=key2: safe_round(
                                    (
                                        meters[key1].sum / meters[key2].sum
                                    ), 5
                                )
                                if meters[key2].sum > 0
                                else float("nan")
                            )

                # for precision / recall
                # elif log_key.endswith("tp"):
                #     log_key = log_key.split("tp")[0]
                #     tps = [log[log_key + "tp"] for log in logging_outputs]
                #     fps = [log[log_key + "fp"] for log in logging_outputs]
                #     fns = [log[log_key + "fn"] for log in logging_outputs]
                #     aggregated_tps = Counter()
                #     aggregated_fps = Counter()
                #     aggregated_fns = Counter()
                #     for tp, fp, fn in zip(tps, fps, fns):
                #         aggregated_tps.update(Counter(tp))
                #         aggregated_fps.update(Counter(fp))
                #         aggregated_fns.update(Counter(fn))
                #     aggregated_tps = dict(aggregated_tps)
                #     aggregated_fps = dict(aggregated_fps)
                #     aggregated_fns = dict(aggregated_fns)
                    
                #     for log_id in aggregated_tps.keys():
                #         key = log_key + str(log_id)
                        
                #         metrics.log_scalar(
                #             "_" + key + "_tp",
                #             aggregated_tps[log_id]
                #         )
                #         metrics.log_scalar(
                #             "_" + key + "_fp",
                #             aggregated_fps[log_id]
                #         )
                #         metrics.log_scalar(
                #             "_" + key + "_fn",
                #             aggregated_fns[log_id]
                #         )
                        
                #         key1 = "_" + key + "_tp"
                #         key2 = "_" + key + "_fp"
                #         key3 = "_" + key + "_fn"
                #         metrics.log_derived(
                #             key + "_precision",
                #             lambda meters, key1=key1, key2=key2: safe_round(
                #                 meters[key1].sum / (meters[key1].sum + meters[key2].sum), 5
                #             )
                #             if (meters[key1].sum + meters[key2].sum) > 0
                #             else float("nan")
                #         )
                #         metrics.log_derived(
                #             key + "_recall",
                #             lambda meters, key1=key1, key3=key3: safe_round(
                #                 meters[key1].sum / (meters[key1].sum + meters[key3].sum), 5
                #             )
                #             if (meters[key1].sum + meters[key3].sum) > 0
                #             else float("nan")
                #         )

                elif log_key.endswith("y_score"):
                    log_key = log_key.split("y_score")[0]
                    y_scores = [log[log_key + "y_score"] for log in logging_outputs]
                    y_trues = [log[log_key + "y_true"] for log in logging_outputs]
                    y_classes = [log[log_key + "y_class"] for log in logging_outputs]

                    log_ids = set()
                    for vals in y_trues:
                        log_ids = log_ids.union(set(vals.keys()))
                    
                    aggregated_scores = {log_id: [] for log_id in log_ids}
                    aggregated_trues = {log_id: [] for log_id in log_ids}
                    aggregated_classes = {log_id: [] for log_id in log_ids}
                    for y_score, y_true, y_class in zip(y_scores, y_trues, y_classes):
                        for log_id in log_ids:
                            if log_id in y_score:
                                aggregated_scores[log_id].append(y_score[log_id])
                                aggregated_trues[log_id].append(y_true[log_id])
                                aggregated_classes[log_id].append(y_class[log_id])

                    for log_id in log_ids:
                        aggregated_scores[log_id] = np.concatenate(aggregated_scores[log_id])
                        aggregated_trues[log_id] = np.concatenate(aggregated_trues[log_id])
                        aggregated_classes[log_id] = np.concatenate(aggregated_classes[log_id])

                        key = log_key + str(log_id)

                        metrics.log_custom(
                            meters.AUCMeter,
                            "_" + key + "_auc",
                            aggregated_scores[log_id],
                            aggregated_trues[log_id],
                            aggregated_classes[log_id]
                        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False