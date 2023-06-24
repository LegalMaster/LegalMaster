# import modules
import argparse
import logging
import random
import pickle
import tqdm
import shortuuid
import pandas as pd
import numpy as np
import os
import bitsandbytes as bn

from utils.model import load_tokenizer_and_model

## models
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

base_model_path = '/home/sojungkim2/legalmaster/7Boutput'
#adapter_path = './adapter/'+['llama_legal', 'llama_chat', 'llama_legal_chat', 'llama_chat_legal'][model_id]
adapter_model_path = '/home/sojungkim2/legalmaster/LegalMaster/ChatAdapterTraining/adapter_chat'

tokenizer, model, _device = load_tokenizer_and_model(base_model_path, adapter_model_path, load_8bit=True)

print(model.state_dict())