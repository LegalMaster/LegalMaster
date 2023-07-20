"""
Evaluation Pipeline for case_hold Multiple Choice task
Usage:
python --
"""

# import modules
from ast import Or
from utils.model import load_tokenizer_and_model, apply_lora_multiple, LlamaForMultipleChoice
from utils.dataset import build_mc_dataset
import sys

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# add classifier for all models
# base llama, llama + legal(mc), llama + legal(ss), llama + chat, llama + legal + chat, llama + chat + legal
def is_LoRA_valid(model):
    _ck = [t for t in model.base_model.model.model.layers[2].mlp.gate_proj.lora_B.parameters()]
    is_valid = float(_ck[0].sum()) != 0
    logger.info(f"LoRA's Weights are {'valid' if is_valid else 'invalid'}")
    if not is_valid:
        sys.exit(0)

    return 
def build_model(base_model_path, lora_path_1, lora_path_2, target_model_path):
    model = LlamaForMultipleChoice.from_pretrained(base_model_path)
    tokenizer, model = apply_lora_multiple(base_model_path, lora_path_1, lora_path_2, target_model_path)

    # check if the loraB's parameters are not all 0
    is_LoRA_valid(model)
    return model


# test with lex glue multiple choice dataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug',
                    action = 'store_true',
                    default = False)
parser.add_argument('--ddp',
                    action = 'store_true',
                    default = False)
parser.add_argument('--batch_size',
                    type = int,
                    default = 32)
parser.add_argument('--base_model_path',
                    type = str,
                    default = 'llama')
parser.add_argument('--lora_path_1',
                    type = str,
                    default = '/path/to/adapter')
parser.add_argument('--lora_path_2',
                    type = str,
                    default = '/path/to/adapter')
parser.add_argument('--target_model_path',
                    type = str,
                    default = '/where/to/save/the/model')


args = parser.parse_args()
args.data_dir = '/home/laal_intern003/LegalMaster/data/case_hold.pkl'
args.eval_result_dir = '/home/laal_intern003/LegalMaster/evalPipeline/results'

# load dataset
eval_dataset = build_mc_dataset(args.data_dir)['test']


# Build DataLoader
from tqdm import tqdm
import torch
from torch.utils.data import(
    TensorDataset,
    DataLoader,
    Dataset
)
import torch.nn.functional as F

def validate(model, dataset):
    pbar = tqdm(total = len, desc = "Iteration")
    model.eval()
    val_dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False)
    logits = []
    labels = dataset['label']

    for batch in pbar(val_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        logger.info(f'batch: {batch}')
        _, _, label, input_ids, attention_mask = batch

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask
            )
        logits.append(outputs.logits)
        logger.info(f'logits: {logits}')
    logits_cat = torch.cat(logits, dim = 0)

    # normalization
    softmax_logits = F.softmax(logits_cat, dim = 1)
    pred = softmax_logits.sum(dim = 0).argmax().item()
    score = softmax_logits.mean(dim = 0).max().item()

    torch.LongTensor(labels)
    is_correct = torch.eq(pred, labels)
    accuracy = round((int(sum(is_correct)) / len(val_dataloader)), 3)
    logger.info(f"\n===== Accuracy: {accuracy} =====\n")

    
    from collections import OrderedDict
    results = OrderedDict()
    results['eval_accuracy'] = accuracy
    results['pred'] = pred
    results['is_correct'] = is_correct
    results['score'] = score

    import os
    import json
    if not os.path.exists(args.eval_result_dir):
        os.makedirs(args.eval_result_dir)
    with open(args.eval_result_dir, 'w') as f:
        json.dump(results, f)
    
    logger.info(f"Results are properely saved at {args.eval_result_dir}")



model = build_model(args.base_model_path, args.lora_path_1, args.lora_path_2, args.target_model_path)
dataset = build_mc_dataset(args.data_dir)

validate(model, dataset)
    
    

