
import builtins
import datetime
import os

import torch
import torch.distributed as dist

import logging
from datetime import datetime
from typing import Sequence
import copy
from  logging import Logger

from omegaconf import DictConfig, OmegaConf
import rich.syntax
import rich.tree
from utils.rank_zero import rank_zero_only

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    nodist = args.nodist if hasattr(args,'nodist') else False 
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and not nodist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=3600))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_logger(cfg, name=None, disable_console=False):
    if disable_console:
        # stop output to stdout; only output to log file
        if 'console' in cfg.job_logging_cfg.root.handlers:
            cfg = copy.deepcopy(cfg)
            cfg.job_logging_cfg.root.handlers.remove('console')
    # log_file_path is used when unit testing
    if is_logging_process():
        logging.config.dictConfig(
            OmegaConf.to_container(cfg.job_logging_cfg, resolve=True)
        )
        return logging.getLogger(name)

from rich.console import Console
from rich.syntax import Syntax

def pretty_print_hydra_config(config: DictConfig) -> None:
    """Parse and resolve a Hydra config, and then pretty-print it."""
    if is_logging_process():
        console = Console()
        config_as_yaml = OmegaConf.to_yaml(config, resolve=True)
        syntaxed_config = Syntax(
            config_as_yaml,
            "yaml",
            theme="ansi_dark",
            indent_guides=True,
            dedent=True,
            tab_size=2,
        )
        console.rule("Current config")
        console.print(syntaxed_config)
        console.rule()

from utils import pylogger
import warnings

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        pretty_print_hydra_config(cfg)

from collections import abc

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def move_to_device(batch, device):
    """ set parameter to gpu or cpu """
    if torch.cuda.is_available():
        if torch.is_tensor(batch):
            return batch.to(device)
        elif isinstance(batch, list):
            return [move_to_device(item, device) for item in batch]
        elif isinstance(batch, dict):
            return {key: move_to_device(value, device) for key, value in batch.items()}
        else:
            return batch
    return batch