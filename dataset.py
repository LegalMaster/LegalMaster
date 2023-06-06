from genericpath import exists
from datasets import load_dataset
import datasets
from tqdm import tqdm
import pickle
import os
import json
from transformers import LlamaForCausalLM, LlamaTokenizer

# load dataset
def _get_dataset(task): # split: train, test, validation

    # define if it is multi-task or not
    # multi_task = True if params_sharing_type == 'hard' else False
    task_list = ['ecthr_a', 'ecthr_b', 'eurlex', 'scotus', 'ledgar', 'unfair_tos', 'case_hold']
    if task not in task_list:
        raise ValueError(f'The task is not in the task_list! {task}')

    dataset = load_dataset('lex_glue', task)

    # # turn the datset into a dict
    # data = {}
    # for split in ['train', 'test', 'validation']:
    #     keys = dataset[f'{split}'].features.keys()
    #     data[f'{split}'] = {}

    #     for key in keys:
    #         data[f'{split}'][f'{key}'] = dataset[f'{split}'][key]

    return dataset

# build dataset
def build_dataset(task_list, path):

    if not os.path.exists(path):
        os.mkdir(path)

    dataset_list = []
    for task in tqdm(task_list):
        dataset = _get_dataset(task)
        dataset_list.append(dataset)

        if not os.path.exists(os.path.join(path, f'{task}.pkl')):
            with open(os.path.join(path, f'{task}.pkl'), 'wb') as f:
                pickle.dump(dataset, f)

    print("Datasets are succesfully loaded")
    return dataset_list


# convert dataset: unsupervised learning task
def _generate_prompt(data_point):
    return data_point["text"]


def _tokenize(prompt, tokenizer, CUTOFF_LEN):
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


def generate_and_tokenize_prompt(data_point, tokenizer, CUTOFF_LEN):
    prompt = _generate_prompt(data_point)
    return _tokenize(prompt, tokenizer, CUTOFF_LEN)


def _answer(data_point):
    return data_point['context'], data_point['endings'][data_point['label']]

def _fill_masked(data_point):
    text, masked = _answer(data_point)
    return {
        "text" : text.replace('(<HOLDING>)', masked)
    }

def rebuild_dataset(dataset):
    datasetdict = {}
    for split in tqdm(['train', 'validation', 'test']):
        datasetdict[split] = dataset[split].map(_fill_masked).select_columns(['text'])
    return datasets.DatasetDict(datasetdict)


# def sampling(data_point, holding_pos):
    
#     data_point
#     pass

# if length > 512:
#     bp
#     ep
# bp
# ep
