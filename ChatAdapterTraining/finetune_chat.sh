python finetune_chat.py \
--model_size 7B \
--batch_size 64 \
--micro_batch_size 32 \
--learning_rate 0.0002 \
--task_list alpaca,stackoverflow,quora \
--data_dir data \
--output_dir adapter_chat_checkpoints
