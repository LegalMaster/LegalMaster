<<<<<<< HEAD:get_answers_chat_legal.sh
python get_answers.py \
--base_model_dir /home/sojungkim2/legalmaster/7Boutput \
--adapter_1_dir  /home/sojungkim2/legalmaster/LegalMaster/ChatAdapterTraining/checkpoints \
--adapter_2_dir  /home/sojungkim2/legalmaster/LegalMaster/LegalAdapterTraining/checkpoints \
--gpu_num 1 \
--model_id 2
=======
#!/bin/bash

#SBATCH --job-name=get_answers_baize
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

source /home/${USER}/.bashrc
conda activate legal-master

srun python get_answers.py --adapter_1_dir /home/laal_intern003/LegalMaster/BaizeAdapter
>>>>>>> d90233032c9ffb1bd78bc7295ee29984b4935a85:get_answers_baize.sh
