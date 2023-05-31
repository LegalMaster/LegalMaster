# import modules
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import socket
from dataset import get_dataset, build_dataset
import torch
import os
import json

# import transformers.adapters.compositoin as ac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        type = str,
                        required = True,
                        help = 'Where the json data is sotred')
    parser.add_argument('--task',
                        action = 'append',
                        nargs= '+',
                        required = True,
                        help = 'The list of datasets you want to download')
    args = parser.parse_args()
    path = args.output_dir
    task_list = args.task[0]
    
    build_dataset(task_list, path)
    # task = 'case_hold'
    
    # with open(os.path.join(path, f'{task}.json')) as f:
    #     dataset = json.load(f)

    # if dataset['train']['context'][0] != None:
    #     print("The dataset is succesfully loaded!")
    


