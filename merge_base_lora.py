"""
Apply the LoRA weights on top of a base model.
Usage:
python merge_lora.py --base ~/model_weights/llama-7b --target ~/model_weights/baize-7b --lora project-baize/baize-lora-7B
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def apply_lora(base_model_path, lora_path_1, target_model_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    print(f"Loading the first LoRA adapter from {lora_path_1}")

    lora_model_1 = PeftModel.from_pretrained(
        base,
        lora_path_1,
        torch_dtype=torch.float16,
    )

    print("Applying the first LoRA adapter to base model")
    model = lora_model_1.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)

def apply_lora_multiple(base_model_path, lora_path_1, lora_path_2, target_model_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    print(f"Loading the first LoRA adapter from {lora_path_1}")

    lora_model_1 = PeftModel.from_pretrained(
        base,
        lora_path_1,
        torch_dtype=torch.float16,
    )

    print("Applying the first LoRA adapter to base model")
    base_new = lora_model_1.merge_and_unload()

    print(f"Loading the first LoRA adapter from {lora_path_2}")

    lora_model_2 = PeftModel.from_pretrained(
        base_new,
        lora_path_2,
        torch_dtype=torch.float16,
    )

    print("Applying the second LoRA adapter to (base + first LoRA adapter) model")
    model = lora_model_2.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_path_1", type=str, required=True)
    parser.add_argument("--lora_path_2", type=str, required=False)
    parser.add_argument("--target_model_path", type=str, required=True)

    args = parser.parse_args()

    if args.lora_path_2:
        apply_lora_multiple(args.base_model_path, args.lora_path_1, args.lora_path_2, args.target_model_path)
    else:
        apply_lora(args.base_model_path, args.lora_path_1, args.target_model_path)
