# import useful old code

import os
import sys
import random
import importlib
import math

import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Any, Dict, Optional, Union

#TODO: configure it
# project_dir = os.path.abspath("/volume/deepecg/fairseq-signals")
# root_dir = project_dir
# if not root_dir in sys.path:
    # sys.path.append(root_dir)

# spec = importlib.util.spec_from_file_location("checkpoint_utils", f"{project_dir}/fairseq_signals/utils/checkpoint_utils.py")
# checkpoint_utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(checkpoint_utils)

from fairseq_signals.utils.checkpoint_utils import load_model_and_task

class WCREcgTransformer(nn.Module):
    def __init__(
        self, 
        model_path: str,
        pretrained_path: str = None,
        overrides: Optional[Dict[str, Any]] = None,
        task=None,
        strict=True,
        suffix="",
        num_shards=1,
        state=None,
    ):
        super().__init__()
        overrides = {} if overrides is None else vars(overrides)
        if pretrained_path is not None:
            overrides.update({"model_path": pretrained_path})
        model, saved_cfg, task = load_model_and_task(
            model_path,
            arg_overrides=overrides,
            suffix=suffix
        )

        self.model = model
        
    def forward(self, x, padding_mask=None):
        net_input = { "source": x, "padding_mask": padding_mask}
        net_output = self.model(**net_input)
        return self.model.get_logits(net_output)




m_root = '/media/data1/achilsowa/results/fairseq/outputs/'

m_paths = {
    "SSL": os.path.join(m_root, "2024-09-22/03-16-32/checkpoints-all/checkpoint_last.pt"),
    "FT_AFIB-5": os.path.join(m_root, "2024-10-17/18-33-35/checkpoint_last-ft-afib-v2-5-bf/checkpoint_best.pt"),
    "FT_FEVG-50": os.path.join(m_root, "2024-10-08/02-30-25/checkpoint_last-ft-fevg-v2-50-bce/checkpoint_best.pt"),
    "FT_FEVG-40": os.path.join(m_root, "2024-10-08/02-31-55/checkpoint_last-ft-fevg-v2-40-bce/checkpoint_best.pt"),
    "FT_FEVG-REG": os.path.join(m_root, "2024-10-17/05-09-15/checkpoint_last-ft-fevg-reg-mse/checkpoint_best.pt"),
    "FT_LABELS-77": os.path.join(m_root, "2024-10-08/04-39-01/checkpoint_last-ft-labels-77-bce/checkpoint_best.pt")
}

# # Get the path to the root directory of your project
# root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # Adjust according to your folder depth

# # Add the root directory to sys.path
# if root_path not in sys.path:
#     sys.path.append(root_path)




def get_model(key):
    return  WCREcgTransformer(m_paths[key], m_paths['SSL'])
