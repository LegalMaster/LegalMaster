#!/bin/bash
#SBATCH --job-name=chat_adaptor
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-10:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=4

source /home/sojungkim2/.bashrc
conda activate legalmaster

srun python finetune_chat.py \
--model_size 7B \
--batch_size 64 \
--micro_batch_size 32 \
--learning_rate 0.0002 \
--task_list alpaca,stackoverflow,quora \
--data_dir data \
--output_dir adapter_chat_test
