#!/bin/bash

#SBATCH --job-name=get_answers_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
source /home/${USER}/.bashrc
conda activate legal-master

srun python -m get_answers