# for distributed learning
import torch.distributed as dist
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# modules
import logging
import socket
import numpy as np

# models
from transformers import (
    LlamaConfig, 
    LlamaForCausalLM, 
    LlamaTokenizer,
    AdamW,
    get_linear_schedule_warmup

)





