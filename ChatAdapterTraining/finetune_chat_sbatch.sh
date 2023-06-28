#!/bin/bash
#SBATCH --job-name=chat_adaptor
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=4

source /home/${USER}/.bashrc
conda activate legal-master

srun python finetune_chat.py \
--model_size 7B \
--batch_size 64 \
--micro_batch_size 16 \
--learning_rate 0.0002 \
--task_list alpaca,stackoverflow,quora \
--data_dir data \
<<<<<<< HEAD
--output_dir checkpoints
=======
--output_dir adapter_chat_checkpoints
>>>>>>> 4987b3e65749e16f939940d8cff4d544f6cb8b34
