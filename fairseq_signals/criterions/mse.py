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
class MSECriterionConfig(Dataclass):
    weight: Optional[List[float]] = field(default = None,metadata = {"help": "For retro compatibility"})
    weights_file: Optional[str] = field(default=None, metadata={"help": "For retro compatibility"})
    per_log_keys: List[str] = field(default_factory = lambda: [],metadata={"help": "For retro compatibility"})
    report_auc: bool = field(default=False,metadata={"help": "For retro compatibility"})
    report_cinc_score: bool = field(default=False,metadata={"help": "For retro compatibility"})
    

@register_criterion(
    "mse2", dataclass = MSECriterionConfig
)
class MSECriterion(BaseCriterion):
    def __init__(self, cfg: MSECriterionConfig, task: Task):
        super().__init__(task)
        

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

        target = target.float()        
        reduction = 'sum' if reduce else 'none'
        loss = F.mse_loss(logits, target, reduction=reduction)

        
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

        metrics.log_scalar("nsignals", nsignals)

        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False