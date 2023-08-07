"""
Get Answers from Llama mounting legal, chat adapters.
Usage:
python get_answers.py \
--model_id 1 \
--num_gpus 2 \
--debug \
"""

# import modules
import argparse
import logging
import random
import pickle
import sklearn
from sklearn import datasets
import tqdm
import shortuuid
import pandas as pd
import numpy as np
import os
import bitsandbytes as bn
from tqdm import tqdm
from datasets import concatenate_datasets

## models
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

## utils
from utils.dataset import build_dataset
from utils.model import load_tokenizer_and_model, load_tokenizer_and_model_multiple, apply_lora_multiple, sample_decode, predict, cleanse_and_split
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# set arugments
parser = argparse.ArgumentParser()
parser.add_argument('--model_id',
                    type = int,
                    default = 0, 
                    help = '0: llama, 1:llama_legal, 2: llama_chat, 3: llama_legal_chat, 4: llama_chat_legal')
parser.add_argument('--num_gpus',
                    type = int,
                    default = 2,
                    help = 'The number of gpus to use for evaluation')
parser.add_argument('--max_seq_len',
                    type = int,
                    default = 100,
                    help = 'Max sequence length of the input text')
parser.add_argument('--task',
                    type = str,
                    default = 'case_hold',
                    help = 'Task: case_hold, ')
parser.add_argument('--adapter_1_dir',
                    type = str,
                    required = False,
                    help = 'Where the adapter 1 is stored')
parser.add_argument('--adapter_2_dir',
                    type = str,
                    required = False,
                    help = 'Where the adapter 2 is stored')
parser.add_argument('--batch_size',
                    type = int,
                    default = 32)
parser.add_argument('--debug',
                    action = 'store_true',
                    default = False)
parser.add_argument('--resume_retrieval',
                    action = 'store_true',
                    default = False)

args = parser.parse_args()
args.data_dir = f'/home/sojungkim2/legalmaster/LegalMaster/data/'
args.question_path = f'/home/sojungkim2/legalmaster/LegalMaster/data/prompt/{args.task}.pkl'
_model_name = ['llama', 'llama_legal', 'llama_chat', 'llama_legal_chat', 'llama_chat_legal']
args.answer_path = f'/home/sojungkim2/legalmaster/LegalMaster/data/answers/{args.model_id}.pkl'
args.base_model_dir = '/home/sojungkim2/legalmaster/7Boutput'
args.save_steps = 500

# dataset

def make_problem(data_point):

    # prompt_cands = [
    # "Please select the most suitable summary of the legal ruling that accompanies the relevant referenced decisions for the specific case. The following excerpt is from the court's decision.",
    # "Kindly choose the concise summary of the legal ruling that accompanies the relevant referenced decisions applicable to the given case. Provided below is an excerpt from the court decision.",
    # "Please decide on the most appropriate summary of the legal ruling that accompanies the relevant referenced decisions, which are relevant to the given case. Here is an excerpt from the court decision for your consideration.",
    # "Here is an excerpt from the court decision for the case. Please choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    # "Consider the following excerpt from the court decision for the case. Your task is to select the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    # "Given the excerpt from the court decision for the case, your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    # "Please refer to the following excerpt from the court decision for the case. Your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    # "Your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case, using the excerpt from the court decision provided below.",
    # "Please review the following excerpt from the court decision for the case. Your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    # "Here is the excerpt from the court decision for the case. Your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case."
    # ] # generated by Chat-GPT

    
    _slice = data_point['context'].find('(<HOLDING>)')
    excerpt = data_point['context'][:_slice]
    choices = [str(n) + f': {c}' for n,c in enumerate(data_point['endings'])]
    # prompt = random.choice(prompt_cands)
    input = f'Excerpt: {excerpt}' + f'\nChoices: {choices}'

    return {
        'question' : input,
        'label' : data_point['label']
    }

def make_dataset(args):
    # build dataset
    data_dir = args.data_dir
    question_path = args.question_path
    task = args.task

    # load file
    if os.path.isfile(question_path):
        with open(question_path, 'rb') as f:
            data = pickle.load(f)
    else:
        dataset = build_dataset(task, data_dir)
        data = dataset['test']
        random.seed(7)
        data = data.shuffle().map(make_problem).select_columns(['question', 'label'])
        data = data.add_column(name = 'idx', column = range(len(data)))

        # save file
        with open(question_path, 'wb') as f:
            pickle.dump(data, f)

        
    return data


def get_model_answers(args, data):   

    # load model
    if args.adapter_2_dir == None:
        tokenizer, model, device = load_tokenizer_and_model(args.base_model_dir, args.adapter_1_dir, load_8bit=True)
    else:
        tokenizer, model, device = load_tokenizer_and_model_multiple(args.base_model_dir, args.adapter_1_dir, args.adapter_2_dir, load_8bit=True)
     
    # arrange data
    print(data)
    questions = data['question']
    labels = data['label']

    answers = []

    # get answers
    for idx, question in enumerate(tqdm(questions)):
        # prompt = question
        # input_ids = tokenizer([prompt], return_tensors = 'pt')['input_ids'][:, -1024:].cuda() # index last 1024 tokens of all questions
        # torch.cuda.empty_cache()
        print(f"{question}")
        response = predict(text = question,
                tokenizer = tokenizer,
                model = model,
                top_p = 0.95,
                temperature = 1.0,
                max_length_tokens = 512,
                max_context_length_tokens = 2048,
                device = device)

        for r in response:
            answer = r[0]
            answer = cleanse_and_split(answer)
            answers.append(answer)

        print("="*80)
        logger.info(f'answers: {answers[idx]}')
        print("Generated")
        # ans_id = shortuuid.uuid()
        # answers.append(
        #     {
        #         "idx" : _idx,
        #         "answer" : outputs,
        #         "answer_id" : ans_id,
        #         # "model_id" : model_id

        #     }
        #     )
        if args.debug:
            break
        
        flag = False
        if idx != 0 and idx%args.save_steps== 0 and flag:
            # update retrieval state
            save_state(idx, args)

            # save answers
            indices = [i for i in range(args.steps, args.steps + args.save_steps)]
            questions = questions[indices[0]: indices[0]+args.save_steps]
            labels = labels[indices[0]: indices[0]+args.save_steps]
            save_answers(args, indices, questions, answers)

    if not args.debug:
        answers = datasets.Dataset.from_pandas(pd.DataFrame({'idx': indices, 'question': questions, 'answer': answers})) # convert to Dataset obj
        with open(args.answer_path, 'wb') as f:
            pickle.dump(data, f)

def save_answers(args, indices, questions, answers):

    data_saved = datasets.Dataset.from_pandas(pd.DataFrame({'idx': indices, 'question': questions, 'answer': answers})) # convert to Dataset obj

    if os.path.isfile(args.answer_path):
        with open(args.answer_path, 'rb') as f:
            prev_data = pickle.load(f)

    data = concatenate_datasets([prev_data, data_saved])
    with open(args.answer_path, 'wb') as f:
        pickle.dump(data, f)
    torch.cuda.empty_cache()

def save_state(idx, args):
    step = 0
    import json
    retrieval_state_path = os.path.join(args.answer_path, 'retrieval_state.json')
    if os.path.isfile(retrieval_state_path):
        with open(retrieval_state_path, 'r') as f:
            step = json.load(f)['step']
    args.step = step + idx

    retrieval_state = {}
    for arg in vars(args):
        retrieval_state[arg] = getattr(args,arg)

    with open(retrieval_state_path, 'w') as f:
        json.dump(retrieval_state, f)

            
def evaluate(args):

    # make dataset
    data = make_dataset(args)
    print("Let's get model's answers")
    get_model_answers(args, data)

    # with open(args.answer_path, 'rb') as f:
    #     answers = pickle.load(f).select_columns(['answer', 'idx'])

    # labels = dataset.select_columns(['label', 'idx'])
    # dataset = datasets.concatenate_datasets([answers, labels])

# def calculate_metric(dataset):

#     for i, data in enumerate(dataset):

#         idx = dataset['idx']
#         answer = dataset['answer']
#         label = dataset['label']

#         incorrect_correct_labels = np.array([0,0,0,0,0,0,0]) # one-hot-encoding: [incorrect, correct, 0,1,2,3,4 (ground truth)]

#         r = int(label in answer) # incorrect: 0, correct: 1
#         incorrect_correct_labels[r] += 1
#         incorrect_correct_labels[label+2] += 1
    
#     return incorrect_correct_labels

# #evaluate(args.model_id, args.data_dir, args.answer_dir, args.gpu_num)


evaluate(args)
