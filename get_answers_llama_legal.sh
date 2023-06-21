#!/bin/bash

#SBATCH --job-name=lexglue
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

source /home/${USER}/.bashrc
conda activate legal-master

srun python get_answers.py --gpu_num 2 --model_id 0