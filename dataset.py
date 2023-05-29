from datasets import load_dataset
from tqdm import tqdm

class LexGlue:
    def __init__(self,task_list):
        self.task_list = task_list
        self.data = dict()
        self.load_data()
    
    def load_data(self):
        for task in tqdm(self.task_list):
            data = load_dataset('lex_glue', task)
            self.data[f'{task}'] = {data}

        return self.data

# load dataset
def get_dataset(params_sharing_type):
    # define if it is multi-task or not
    multi_task = True if params_sharing_type == 'hard' else False

    if multi_task:
        task_list = ['ecthr_a', 'ecthr_b', 'eurlex', 'scotus', 'ledgar', 'unfair_tos', 'case_hold']
    else:
        task_list = ['case_hold'] # QA datset

    dataset = LexGlue(task_list)
    return dataset

# convert dataset
def convert_dataset(dataset):
    pass