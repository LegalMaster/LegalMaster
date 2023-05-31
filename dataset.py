from genericpath import exists
from datasets import load_dataset
from tqdm import tqdm
import pickle
import os
import json

# load dataset
def get_dataset(task): # split: train, test, validation

    # define if it is multi-task or not
    # multi_task = True if params_sharing_type == 'hard' else False
    task_list = ['ecthr_a', 'ecthr_b', 'eurlex', 'scotus', 'ledgar', 'unfair_tos', 'case_hold']
    if task not in task_list:
        raise ValueError(f'The task is not in the task_list! {task}')

    dataset = load_dataset('lex_glue', task)

    # turn the datset into a dict

    data = {}
    for split in ['train', 'test', 'validation']:
        keys = dataset[f'{split}'].features.keys()
        data[f'{split}'] = {}

        for key in keys:
            data[f'{split}'][f'{key}'] = dataset[f'{split}'][key]

    return data

# build dataset
def build_dataset(task_list, path):

    if not os.path.exists(path):
        os.mkdir(path)

    for task in tqdm(task_list):
        dataset = get_dataset(task)
        with open(os.path.join(path, f'{task}.json'), 'w') as f:
            json.dump(dataset, f)

    print("Datasets are succesfully loaded")


# convert dataset
def convert_dataset(dataset):
    pass