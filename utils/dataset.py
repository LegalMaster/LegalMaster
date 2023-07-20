from genericpath import exists
from datasets import load_dataset
import datasets
from tqdm import tqdm
import pickle
import os
import json
from transformers import LlamaForCausalLM, LlamaTokenizer

# build dataset
def build_dataset(task, path):

    if not os.path.exists(path):
        os.mkdir(path)

    if os.path.isfile(os.path.join(path, f'{task}.pkl')):
        with open(os.path.join(path, f'{task}.pkl'), 'rb') as f:
            dataset = pickle.load(f)
    else:
        # load dataset
        def get_dataset(task): # split: train, test, validation

            # define if it is multi-task or not
            # multi_task = True if params_sharing_type == 'hard' else False
            task_list = ['ecthr_a', 'ecthr_b', 'eurlex', 'scotus', 'ledgar', 'unfair_tos', 'case_hold']
            if task not in task_list:
                raise ValueError(f'The task is not in the task_list! {task}')

            dataset = load_dataset('lex_glue', task)
            return dataset

        dataset = get_dataset(task)
        with open(os.path.join(path, f'{task}.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

    print("Datasets are succesfully loaded")
    return dataset

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
        "attention_mask": result["attention_mask"][:-1], # paddig
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
    path = '/home/laal_intern003/LegalMaster/data/case_hold_unmasked.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            dataset_rebuilt = pickle.load(f)
    else:
        datasetdict = {}
        for split in tqdm(['train', 'validation', 'test']):
            datasetdict[split] = dataset[split].map(_fill_masked).select_columns(['text'])
            dataset_rebuilt = datasets.DatasetDict(datasetdict)
        with open(path, 'wb') as f:
            pickle.dump(dataset_rebuilt, f)

    return dataset_rebuilt


import pickle
def build_mc_dataset(data_dir):
    # load dataset
    if os.path.isfile(data_dir):
        with open(data_dir, 'rb') as f:
            data_mapped = pickle.load(f)
    else:    
        with open(data_dir, 'rb') as f:
            data = pickle.load(f)

        def preprocess_function(examples):
            context_sentences = [[context] * 5 for context in examples["context"]]
            ending_sentences = [endings for endings in examples['endings']]
            print(context_sentences)
            print(ending_sentences)
            context_sentences = sum(context_sentences, [])
            ending_sentences = sum(ending_sentences, [])

            tokenized_examples = tokenizer(context_sentences, ending_sentences, truncation=True)
            return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}


        data_mapped = data.map(preprocess_function, batched = True)
        with open(data_dir, 'wb') as f:
            pickle.dump(data_mapped, f)
    return data_mapped