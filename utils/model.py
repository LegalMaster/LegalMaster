# import modules
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import transformers
from typing import Iterator
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def load_tokenizer_and_model(base_model_path, adapter_path, load_8bit=False):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map = {"":0}
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model_path, device_map={"": device}, low_cpu_mem_usage=True
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                device_map={"": device},
            )

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.eval()
    return tokenizer, model, device

def load_tokenizer_and_model_multiple(base_model_path, adapter_1_path, adapter_2_path, load_8bit=False):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    if device == "cuda":
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        if adapter_1_path is not None:
            new_base_model = PeftModel.from_pretrained(
                base_model,
                adapter_1_path,
                torch_dtype=torch.float16,
                device_map = {"":0}
            )
            if adapter_2_path is not None:
                merged_model = PeftModel.from_pretrained(
                    new_base_model,
                    adapter_2_path,
                    torch_dtype=torch.float16,
                    device_map = {"":0}
            )
    elif device == "mps":
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if adapter_1_path is not None:
            new_base_model = PeftModel.from_pretrained(
                base_model,
                adapter_1_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            if adapter_2_path is not None:
                merged_model = PeftModel.from_pretrained(
                    new_base_model,
                    adapter_2_path,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
    else:
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path, device_map={"": device}, low_cpu_mem_usage=True
        )
        if adapter_1_path is not None:
            new_base_model = PeftModel.from_pretrained(
                base_model,
                adapter_1_path,
                device_map={"": device},
            )
            if adapter_2_path is not None:
                merged_model = PeftModel.from_pretrained(
                    new_base_model,
                    adapter_2_path,
                    device_map={"": device},
                )

    if not load_8bit and device != "cpu":
        merged_model.half()  # seems to fix bugs for some users.

    merged_model.eval()
    return tokenizer, merged_model, device

# merge model
def apply_lora_multiple(base_model_path, lora_path_1, lora_path_2, target_model_path):
    print(f"Loading the base model from {base_model_path}")
    base = LlamaForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_fast=False)

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

    return base_tokenizer, model


# answer generation
from typing import Iterator

def sample_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    # stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 25,
) -> Iterator[str]:
    generated_tokens = []
    past_key_values = None
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature
        probs = torch.softmax(logits, dim=-1)

        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # -1: last dimension
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # # apply top_k
        # if top_k is not None:
        #    probs_sort1, _ = torch.topk(probs_sort, top_k)
        #    min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
        #    probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)
        # print(f'input_ids: {input_ids}')

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)
        # print(text)
        yield text
        
def predict(text,
    tokenizer,
    model,
    #history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    device,):

    prompt = "You are Legal Master, an AI chatbot. Solve the following legal multiple-choice problem. The answer should be the label with corresponding text."
    prompt = "Solve the following legal multiple-choice problem. The answer should be the label with corresponding text."
    print(prompt)
    inputs = tokenizer(prompt + text, return_tensors = "pt")

    begin_length = len(prompt)

    input_ids = inputs['input_ids'][:, -max_context_length_tokens:].to(device) # max history tokens
    torch.cuda.empty_cache()

    with torch.no_grad():
        for x in sample_decode(
            input_ids,
            model,
            tokenizer,
            max_length = max_length_tokens,
            temperature = temperature,
            top_p = top_p,
        ):
            continue
            # logger.info(x)

    torch.cuda.empty_cache()

    # print(f'Prompt: {prompt}')
    # print(f'Generated Text: {x}')
    # print("="*80)

    try:
        yield x, "Generate: Success"
    except:
        pass

    return prompt

def cleanse_and_split(answer: str, stop_words = ['</s>', '<s>']):
    for word in stop_words:
        answer = answer.replace(word, '')

    return answer

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


# LlamaForMultipleChoice
# import modules
from transformers import LlamaPreTrainedModel, LlamaModel, LlamaTokenizer
from transformers.utils import ModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

# model
@dataclass 
class MultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class LlamaForMultipleChoice(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.model = LlamaModel(config)
        self.num_choices = 5
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights and apply final processing
        self.post_init()
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_iput_embeddings(self, value):
        self.model.embed_tokens = value
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)"""
        
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        logger.info(f'input_ids shape: {input_ids.shape}')
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.model(
            input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logger.info(f'hidden_states shape: {hidden_states.shape}')
        logits = self.classifier(hidden_states)
        logger.info(f'logits shape: {logits.shape}')
    
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes >1 if no padding token is defined")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) -1).to(logits.device)
            else:
                sequence_lengths = -1
        
        pooled_logits = logits[torch.arange(batch_size, device = logits.device), sequence_lengths].transpose(0,1)
        logger.info(f'pooled logits shape: {pooled_logits.shape}')

    
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.num_choices == 1:
                raise ValueError("Multiple Choice Task requires at least two labels but only one is given")
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_choices), labels.view(-1))
        else:
            raise ValueError("Multiple Choice Task requires at least two labels but nothing is given")
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return MultipleChoiceModelOutput(
            loss = loss,
            logits = pooled_logits,
            # hidden_states = outputs.hidden_states,
            # attentions = outputs.attentions,
        )


