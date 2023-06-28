"""
Finetune the LoRA model for chat dataset, with LLaMA 7B base model.
Usage:
python finetune_chat.py \
--model_size 7B \
--batch_size 64 \
--micro_batch_size 32 \
--learning_rate 0.0002 \
--task_list alpaca,stackoverflow,quora \
--data_dir data \
--output_dir adapter_chat

"""
# import modules: utils
import os
import sys
import torch
import pickle
import random
import json
import bitsandbytes as bnb
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

def main(args):
    # parameters
    MICRO_BATCH_SIZE = args.micro_batch_size
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    model_size = args.model_size # The size of Llama (i.e.,7B, 13B, 30B)
    EPOCHS = 1
    LEARNING_RATE = args.learning_rate
    CUTOFF_LEN = 512
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    VAL_SET_SIZE = 2000 # The size of validation set
    TARGET_MODULES = [
        "q_proj", # query projection
        "k_proj", # key projection
        "v_proj", # value projection
        "down_proj", # down projection
        "gate_proj", # gate projection
        "up_proj", # up projection
    ]
    # DATA_PATH = "data/data_tmp.json"
    # OUTPUT_DIR = "checkpoints/{}".format(size)

    # load data
    data = []
    for x in args.task_list.split(","):
        data += json.load(open("{}/{}_chat_data.json".format(args.data_dir,x)))
    random.shuffle(data)
    json.dump(data, open("{}/data_tmp.json".format(args.data_dir), "w"))
    data = load_dataset("json", data_files="{}/data_tmp.json".format(args.data_dir))
    # print(data)

    # load model
    device_map = "auto"
    # print("World size:", str(os.environ.get("WORLD_SIZE")))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1 # If trying to use more than one GPU, change this number  
    
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # GPU's ID being used
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    model = LlamaForCausalLM.from_pretrained(
        "/home/laal_intern003/LegalMaster/llama/",
        load_in_8bit=True, # This option saves memory
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        "/home/laal_intern003/LegalMaster/llama/", 
        add_eos_token=True # add end-of-sentence token
    )

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=LORA_R, # determines the compression rate
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
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
        "Total number of parameters: {}M, rate: {}%".format(
            total_params // 1000 / 1000, round(total_params / params * 100, 2)
        )
    )


    # Data Preprocess
    def generate_prompt(data_point):
        return data_point["input"]


    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }


    def generate_and_tokenize_prompt(data_point):
        prompt = generate_prompt(data_point)
        return tokenize(prompt)


    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    # with open('./data/train_data.pkl', 'wb') as f:
    #     pickle.dump(train_data, f)


    # Training Setting
    trainer = transformers.Trainer(
        model = model,
        train_dataset = train_data,
        eval_dataset = val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size = MICRO_BATCH_SIZE,
            per_device_eval_batch_size = MICRO_BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            warmup_steps = 100,
            num_train_epochs = EPOCHS,
            learning_rate = LEARNING_RATE,
            fp16 = True,
            logging_steps = 20,
            evaluation_strategy = "steps" if VAL_SET_SIZE > 0 else "no",
            save_strategy = "steps",
            eval_steps = 200 if VAL_SET_SIZE > 0 else None,
            save_steps = 200,
            output_dir = args.output_dir,
            save_total_limit = 100,
            load_best_model_at_end = True if VAL_SET_SIZE > 0 else False,
            ddp_find_unused_parameters = False if ddp else None
            ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    

    trainer.train()
    #model.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size',
                        type = str,
                        default = '7B',
                        help = 'Foundation model size (i.e., 7B, 13B, 30B)')
    parser.add_argument('--batch_size',
                        type = int,
                        default = 64,
                        help = 'Batch size') # The bigger, the faster
    parser.add_argument('--micro_batch_size',
                        type = int,
                        default = 16,
                        help = 'Micro Batch size') # The bigger, the faster
    parser.add_argument('--learning_rate',
                        type = float,
                        default = 0.0002,
                        help = 'Learning rate')
    parser.add_argument('--task_list',
                        type = str,
                        help = 'The string of datasets you want to download, seperated by commas')
    parser.add_argument('--data_dir',
                        type = str,
                        default = 'data',
                        help = 'Where dataset is stored')
    parser.add_argument('--output_dir',
                        type = str,
                        default = 'checkpoints',
                        help = 'Where the trained adapter is stored')
    
    args = parser.parse_args()
    main(args)