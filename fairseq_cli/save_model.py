import logging
import os
import sys
from itertools import chain
import pprint

from omegaconf import DictConfig

import numpy as np
import torch
import dill
import io

from fairseq_signals import distributed_utils
from fairseq_signals.utils import checkpoint_utils, options, utils
from fairseq_signals.dataclass.initialize import add_defaults, hydra_init
from fairseq_signals.dataclass.utils import omegaconf_no_object_check
from fairseq_signals.logging import progress_bar
from fairseq_signals.dataclass.configs import Config
from fairseq_signals.utils.utils import reset_logging
from fairseq_signals.utils.store import initialize_store, store

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.save_model")

def main(cfg: DictConfig, override_args=None):
    #torch.multiprocessing.set_sharing_strategy("file_system")

    utils.import_user_module(cfg.common)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
    else:
        overrides = {}

    overrides.update({"task": {"data": cfg.task.data}})
    model_overrides = eval(getattr(cfg.common_eval, "model_overrides", "{}"))
    overrides.update(model_overrides)

    # Load model
    logger.info(f"loading model from {cfg.common_eval.path}")

    logger.info('*'*10, f'override: {overrides}', '--------------', cfg.checkpoint.checkpoint_suffix)
    model, saved_cfg, task = checkpoint_utils.load_model_and_task(
        cfg.common_eval.path,
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix
    )
    save_path = os.path.join(os.path.dirname(cfg.common_eval.path), 'model.pt')

    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad)
        )
    )

    # Move model to GPU
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    with open(save_path, 'wb') as file:
        torch.save(model, file, pickle_module=dill)
        #dill.dump(model, file)

    dummy_input = torch.randn(1, 12, 2500).to("cuda")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(save_path)
    model = torch.jit.load(save_path)
    #with open(save_path, 'rb') as file:
        #rmodel = dill.load(file)
    #model = torch.load(save_path, pickle_module=dill)

    logger.info(f"saving model to {save_path}")
    
    
    
def cli_main():
    parser = options.get_inference_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(
        override_parser, suppress_defaults = True
    )

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )

if __name__ == "__main__":
    cli_main()
