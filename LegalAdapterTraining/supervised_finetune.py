"""
Finetune the LoRA model for Lexglue dataset, with LLaMA 7B base model based on Multiple Choice Selection Task

Usage:
python finetune.py --task_list case_hold 

"""
# import modules
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size',
                        type = str,
                        default = '7B',
                        help = 'Foundation model size (i.e., 7B, 13B, 30B)')
    parser.add_argument('--micro_batch_size',
                        type = int,
                        default = 16,
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
                        default = 'data',
                        help = 'Where LexGlue dataset is stored')
    parser.add_argument('--output_dir',
                        type = str,
                        default = './checkpoints',
                        help = 'Where the adapter is stored')
    parser.add_argument('--debug',
                        action = 'store_true',
                        default = False,
                        help = 'debug mode')
    parser.add_argument('--num_gpus',
                        type = int,
                        default = 4,
                        help = 'The number of gpus to be used')