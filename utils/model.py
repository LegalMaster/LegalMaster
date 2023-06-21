# import modules
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import transformers
from typing import Iterator

def load_tokenizer_and_model(base_model, adapter_model, load_8bit=False):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model,
                device_map={"": device},
            )

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.eval()
    return tokenizer, model, device

def sample_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    # stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 25,
) -> str:
    generated_tokens = []
    past_key_values = None
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            # print(f'outputs: {outputs}, logits: {logits}') # test
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # apply top_k
        if top_k is not None:
           probs_sort1, _ = torch.topk(probs_sort, top_k)
           min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
           probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)
        # print(f'input_ids: {input_ids}')

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)
        print(text)

    print(generated_tokens)

    return generated_tokens
        # if any([x in text for x in stop_words]):
        #     return

def simple_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    # stop_words: list,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 25,
) -> str:
    print(f'length of the input_ids: {len(input_ids[0])}')
    generate_ids = model.generate(input_ids, max_new_tokens = max_new_tokens)[:, len(input_ids[0]):]
    print(f'{len(generate_ids) == len(generate_ids[:,len(input_ids):])}')
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
    print(text)
    return text