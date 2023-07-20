from utils.model import load_tokenizer_and_model
import torch
import logging
import time


pst_time = time.time()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

base_model = '/home/sojungkim2/legalmaster/7Boutput'
adapter_model = '/home/sojungkim2/legalmaster/LegalMaster/ChatAdapterTraining/adapter_chat_test_2/checkpoint-10'
tokenizer, model, _device = load_tokenizer_and_model(base_model, adapter_model, load_8bit=True)
print([t for t in model.base_model.model.model.layers[3].self_attn.q_proj.lora_B.parameters()])
#print(model.state_dict())
crr_time = time.time()

print(f'time spent: {crr_time - pst_time}')