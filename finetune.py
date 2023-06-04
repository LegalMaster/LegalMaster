# import modules: utils
from calendar import EPOCH
import os
import sys
import pickle
import random
import json
import bitsandbytes as bn
import argparse

# import modules: learning
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# import modules: dataset
from datasets import load_dataset
from dataset import *



def main(args):
    # parameters
    MICRO_BATCH_SIZE = args.micro_batch_size
    BATCH_SIZE = 64
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    model_size = args.model_size # The size of Llama (i.e.,7B, 13B, 30B)
    EPOCHS = 1
    LEARNING_RATE = args.learning_rate
    CUTOFF_LEN = 512
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    # VAL_SET_SIZE = 2000 -- we don't need this cuz LexGLUE is already splited into train/val/train
    TARGET_MODULES = [
        "q_proj", # query projection
        "k_proj", # key projection
        "v_proj", # value projection
        "down_proj", # down projection
        "gate_proj", # gate projection
        "up_proj"   # up projection
    ]

    # load data
    task_list = args.task_list[0]
    path = args.data_dir
    dataset_list = build_dataset(task_list, path)
    dataset = dataset_list[0]
    print("Datasets are loaded") # dict, {task_name:data}

    # load model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE",1))
    ddp = world_size != 1 # if there are multiple GPUs, we will do distributed learning
    
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # process id for the corresponding gpu
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
    
    model = LlamaForCausalLM.from_pretrained(
        './llama',
        load_in_8bit = True, # this saves memory
        device_map = device_map)

    tokenizer = LlamaTokenizer.from_pretrained(
        './llama',
        add_eos_token = True # add end-of-sentence token
    )
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r = LORA_R, # determines the compression rate
        lora_alpha = LORA_ALPHA,
        target_modules = TARGET_MODULES,
        lora_dropout = LORA_DROPOUT,
        bias = "none",
        task_type = "CAUSAL_LM"
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    config.save_pretrained(args.output_dir)

    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0

    # count the total nums of params in the model
    total_params, params = 0, 0 
    for n, p in model.model.named_parameters():
        if any([x in n for x in ["lora"]]):
            total_params += p.numel()
        params += p.numel()
    print(
        f'Total number of parameters:{total_params // 1000 / 1000}M, rate {round(total_params / params * 100, 2)}%'
    )

    # train
    # split data
    train_data = dataset['train'].shuffle().map(lambda data: generate_and_tokenize_prompt(data, tokenizer, CUTOFF_LEN)).select_columns(['input_ids', 'attention_mask'])
    val_data = dataset['validation'].shuffle().map(lambda data: generate_and_tokenize_prompt(data, tokenizer, CUTOFF_LEN)).select_columns(['input_ids', 'attention_mask'])
    test_data = dataset['test'].shuffle().map(lambda data: generate_and_tokenize_prompt(data, tokenizer, CUTOFF_LEN)).select_columns(['input_ids', 'attention_mask'])
    
    # select features
    trainer = transformers.Trainer(
        model = model,
        train_dataset = train_data,
        eval_dataset = val_data,
        args = transformers.TrainingArguments(
            per_device_train_batch_size = MICRO_BATCH_SIZE,
            per_device_eval_batch_size = MICRO_BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            warmup_steps = 100,
            num_train_epochs = EPOCHS,
            learning_rate = LEARNING_RATE,
            fp16 = True,
            logging_steps = 20,
            evaluation_strategy = "steps",
            save_strategy = "steps",
            eval_steps = 200,
            save_steps = 200,
            output_dir = args.output_dir,
            save_total_limit = 100,
            load_best_model_at_end = True,
            ddp_find_unused_parameters = False if ddp else None
            ),
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False)
        )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__ : get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size',
                        type = str,
                        default = '7B',
                        help = 'Foundation model size (i.e., 7B, 13B, 30B)')
    parser.add_argument('--micro_batch_size',
                        type = int,
                        default = 32,
                        help = 'Batch size') # The bigger, the faster
    parser.add_argument('--learning_rate',
                        type = float,
                        default = 0.0002,
                        help = 'Learning rate')
    parser.add_argument('--task_list',
                        action = 'append',
                        nargs= '+',
                        required = True,
                        help = 'The list of datasets you want to download')
    parser.add_argument('--data_dir',
                        type = str,
                        default = './dataset',
                        help = 'Where LexGlue dataset is stored')
    parser.add_argument('--output_dir',
                        type = str,
                        default = './checkpoints',
                        help = 'Where the adapter is stored')
    
    args = parser.parse_args()
    main(args)

