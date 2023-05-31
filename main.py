# import modules
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import socket
from dataset import LexGlue, get_dataset, build_dataset
import torch

#import transformers.adapters.composition as ac

def main(args):
    
    # load lex_glue dataset
    task_list = args.task_list
    dataset = build_dataset(args.params_sharing_type)

    print("Datasets are loaded") # dict, {task_name:data}

    # train
    # load llama
    model = LlamaForCausalLM.from_pretrained("./llama")
    tokenizer = LlamaTokenizer.from_pretrained("./llama")
    model.add_adapter('LLaMA-Adapter')
    print("adapter added")

    # validate

    # test
    


if __name__ == "__main__":
    print(f'Job is running on {socket.gethostname()}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_sharing_type',
                        default = 'hard',
                        type = str,
                        help = 'Multi-tasking parameter-tuning type: hard/soft' )
    parser.add_argument('--non_parallel',
                        default=False,
                        action = 'store_true',
                        help = 'Do not use multiprocessing')
    parser.add_argument('--cache_dir',
                        default = './data/models/',
                        type = str,
                        help = 'Where the pre-trained models will be / is stored')
    parser.add_argument('--checkpoint_dir',
                        default = './best_ckpt/',
                        type = str,
                        help = 'Where the best checkpoint for LLaMa is stored')
    parser.add_argument('--checkpoint',
                        default = 'best_ckpt.pth',
                        type = str,
                        help = 'The best chceckpoint for LLaMa model')
    parser.add_argument('--batchsize',
                        defualt = 8,
                        type = int,
                        help = 'Batch size for validation examples')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.parallel = ~(args.non_parallel)
    args.n_cpu = args.n_cpu if args.parallel else 1

    main(args)



