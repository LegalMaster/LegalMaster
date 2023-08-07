# import modules
import argparse
import logging
import random
import pickle
import ray
#from sklearn 
import datasets
import tqdm
import shortuuid
import pandas as pd
import numpy as np
import os
import bitsandbytes as bn
from tqdm import tqdm

## models
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

## utils
from utils import dataset, model
from utils.model import sample_decode, load_tokenizer_and_model, load_tokenizer_and_model_multiple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
  
        
# pipeline
def make_dataset(data_dir):
    # build dataset
    path = data_dir

    # load file
    if os.path.isfile(os.path.join(path, 'prompt.pkl')):
        with open(os.path.join(path, 'prompt.pkl'), 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset_list = build_dataset(['case_hold'], path)
        dataset = dataset_list[0]['test']
        random.seed(7)
        dataset = dataset.shuffle().map(prompt_engineering).select_columns(['question', 'label'])
        dataset = dataset.add_column(name = 'idx', column = range(3600))
        # save file
        with open(os.path.join(path, 'prompt.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

    return dataset

def prompt_engineering(data_point):

    prompt_cands = [
    "Please select the most suitable summary of the legal ruling that accompanies the relevant referenced decisions for the specific case. The following excerpt is from the court's decision.",
    "Kindly choose the concise summary of the legal ruling that accompanies the relevant referenced decisions applicable to the given case. Provided below is an excerpt from the court decision.",
    "Please decide on the most appropriate summary of the legal ruling that accompanies the relevant referenced decisions, which are relevant to the given case. Here is an excerpt from the court decision for your consideration.",
    "Here is an excerpt from the court decision for the case. Please choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    "Consider the following excerpt from the court decision for the case. Your task is to select the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    "Given the excerpt from the court decision for the case, your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    "Please refer to the following excerpt from the court decision for the case. Your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    "Your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case, using the excerpt from the court decision provided below.",
    "Please review the following excerpt from the court decision for the case. Your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case.",
    "Here is the excerpt from the court decision for the case. Your task is to choose the most appropriate short summary of the legal ruling that accompanies the referenced decisions relevant to the case."
    ] # generated by Chat-GPT

    
    _slice = data_point['context'].find('(<HOLDING>)')
    excerpt = data_point['context'][:_slice]
    choices = [str(n) + f': {c}' for n,c in enumerate(data_point['endings'])]
    prompt = random.choice(prompt_cands)
    input = prompt + f'\nExcerpt: {excerpt}' + f'\nChoices: {choices}'

    return {
        'question' : input,
        'label' : data_point['label']
    }

class ModelIDAdapterMismatchError(Exception):
    pass

#def run_generate(model_id, dataset, answer_path, num_gpus = 3):
def run_generate(model_id, dataset, answer_path):
    questions = dataset.select_columns(['question', 'idx', 'label'])

    # chunk_size = len(questions) // num_gpus
    # ans_handlers = []
    
    # tokenizer, model, _device = load_tokenizer_and_model(args.base_model_dir, args.adapter_1_dir, load_8bit=True)
    if torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"

    try:
        if torch.backends.mps.is_available():
            _device = "mps"
    except:  # noqa: E722
        pass
    
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model_dir)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model_dir,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    if args.adapter_1_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_1_dir,
            torch_dtype=torch.float16,
            device_map = {"":0}
        )
    if args.adapter_2_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_2_dir,
            torch_dtype=torch.float16,
            device_map = {"":0}
    )
    model.eval()
    
    # print(model_id)
    # if model_id in [0, 1]:
    #     if args.adapter_2_dir:
    #         raise ModelIDAdapterMismatchError("For model{}, your adapter_2_dir argument should be empty, since only one adapter is necessary".format(model_id))
    #     else:
    #         tokenizer, model, _device = load_tokenizer_and_model(args.base_model_dir, args.adapter_1_dir, load_8bit=True)
    # if model_id in [2, 3]:
    #     if not args.adapter_2_dir:
    #         raise ModelIDAdapterMismatchError("For model{}, your adapter_2_dir argument should be given, since 2 adapters are necessary".format(model_id))
    #     else:
    #        tokenizer, model, _device = load_tokenizer_and_model_multiple(args.base_model_dir, args.adapter_1_dir, args.adapter_2_dir, load_8bit=True)        
        
    device_map = {"":0}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
#     ddp = world_size != 1
#     if ddp:
#         device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
#         GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
 
    #model = LlamaForCausalLM.from_pretrained('llama', device_map = device_map, torch_dtype = torch.float16)

    print(f'Device: {_device}')
    answers = get_model_answers(tokenizer, model, questions, device_map)

    # with torch.no_grad():
    #     for question in questions:
    #         answers.append(get_model_answers(tokenizer, model, question))
    # answers = pd.DataFrame(answers)
    
    # answers = datasets.Dataset.from_pandas(answers) # convert to Dataset obj

    if not os.path.exists(answer_path):
        os.makedirs(answer_path)
    # save the answer
    answers.to_csv(os.path.join(answer_path, f'answers_{model_id}.csv'), index = False )
    # with open(os.path.join(answer_path, f'answers_{model_id}.pkl'), 'wb') as f:
    #     pickle.dump(answers, f)
    torch.cuda.empty_cache()


def get_model_answers(tokenizer, model, questions, device_map):

    answers = pd.DataFrame(columns = ["prompt_id", "prompt","answer_id", "answer", "label"])

    with torch.no_grad(): # inactivates pytorch autograd engine so that gradient is not tracked anymore, saving memory and accelerating speed

        for _idx, question in enumerate(tqdm(questions)):
            if _idx > 10: 
                break
            prompt = question['question']
            label = question['label']
            input_ids = tokenizer([prompt], return_tensors = 'pt')['input_ids'][:, -1024:].cuda() # index last 1024 tokens of all questions
            torch.cuda.empty_cache()

#             print(f'question: {question}')

#             outputs = simple_decode(
#                 input_ids,
#                 model,
#                 tokenizer,
#                 max_new_tokens = 50,
#             )
            outputs = sample_decode(
                        input_ids,
                        model,
                        tokenizer,
                        max_length = 100,)
            print(outputs)

            ans_id = shortuuid.uuid()
            print(answers)
            answers = answers.append(
                {
                    "prompt_id" : _idx,
                    "prompt": prompt, 
                    "answer_id" : ans_id,
                    "answer" : outputs,
                    "label" : label
                },
                ignore_index = True
                )

    return answers
            

# def evaluate(model_id, data_dir, answer_path, num_gpus):
def evaluate(model_id, data_dir, answer_path):
    # define model

    #model_name = ['llama_legal', 'llama_chat', 'llama_chat_legal', 'llama_legal_chat'][model_id]

    # get answers
    dataset = make_dataset(data_dir)
    # run_generate(model_id, dataset, answer_path, num_gpus)
    run_generate(model_id, dataset, answer_path)

    with open(os.path.join(answer_path, f'answers_{model_id}.csv'), 'rb') as f:
        answers = pickle.load(f).select_columns(['answer', 'idx'])
    
    labels = dataset.select_columns(['label', 'idx'])

    dataset = datasets.concatenate_datasets([answers, labels])

@ray.remote
def calculate_metric(dataset):

    for i, data in enumerate(dataset):

        idx = dataset['idx']
        answer = dataset['answer']
        label = dataset['label']

        incorrect_correct_labels = np.array([0,0,0,0,0,0,0]) # one-hot-encoding: [incorrect, correct, 0,1,2,3,4 (ground truth)]

        r = int(label in answer) # incorrect: 0, correct: 1
        incorrect_correct_labels[r] += 1
        incorrect_correct_labels[label+2] += 1
    
    return incorrect_correct_labels


#def chatgpt_get_answer(gpt_key):


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type = str,
                        default = './data/prompt',
                        help = 'Where LexGlue dataset is stored')
    parser.add_argument('--answer_dir',
                        type = str,
                        default = './data/answers',
                        help = 'Where answers from the model will be stored')
    parser.add_argument('--base_model_dir',
                        type = str,
                        #default = './llama',
                        help = 'Where base model is stored')
    parser.add_argument('--adapter_1_dir',
                        type = str,
                        required = False,
                        help = 'Where the first adapter is stored')
    parser.add_argument('--adapter_2_dir',
                        type = str,
                        required = False,
                        help = 'Where the second adapter is stored')
    parser.add_argument('--gpu_num',
                        type = int,
                        default = 2,
                        help = 'The number of gpus to use for evaluation')
    parser.add_argument('--model_id',
                        type = int,
                        default = 0, 
                        help = '0: llama_legal, 1: llama_chat, 2: llama_chat_legal, 3: llama_legal_chat')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)

    # generated = evaluate(args.model_id, args.data_dir, args.answer_dir, args.gpu_num)
    evaluate(args.model_id, args.data_dir, args.answer_dir)