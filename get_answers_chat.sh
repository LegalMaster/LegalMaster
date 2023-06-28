: <<'END'
#!/bin/bash

#SBATCH --job-name=chat_adapter_generate
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

source /home/${USER}/.bashrc
conda activate legalmaster

srun python get_answers.py \
--base_model_dir /home/sojungkim2/legalmaster/7Boutput \
--adapter_1_dir  /home/sojungkim2/legalmaster/LegalMaster/ChatAdapterTraining/adapter_chat_checkpoints \
--gpu_num 4 \
--model_id 1

END

python get_answers.py \
--base_model_dir /home/sojungkim2/legalmaster/7Boutput \
--adapter_1_dir  /home/sojungkim2/legalmaster/LegalMaster/ChatAdapterTraining/adapter_chat_checkpoints \
--gpu_num 4 \
--model_id 1