from itertools import chain
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, LlamaForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Callable, List, Literal, Optional, Type, TypeVar, Union
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from utils.llama_contrastive_skip_model import batch_skip_decoder_layer_hook
import warnings
import os
  

class ContrastiveDecodeWithDifferentPromptModel(PreTrainedModel):
    def __init__(
        self, expert: LlamaForCausalLM, amateur: LlamaForCausalLM, alpha: float, beta: float
    ) -> None:
        expert.config._attn_implementation = 'eager'
        super().__init__(expert.config)
        self.expert = expert
        self.amateur = amateur
        self.alpha = alpha
        self.beta = beta
        self.dynamic_beta_scale = False
        self.amateur_input_ids_transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
        self.amateur_mask_layer: Optional[tuple[int, int]] = None
        self.cd_condition: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = \
            lambda full_expert_input_ids, scores: torch.ones(
                full_expert_input_ids.size(0), dtype=torch.bool, device=full_expert_input_ids.device)
        self.use_ensemble = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        assert input_ids.size(0) % 2 == 0
        real_batch_size = input_ids.size(0) // 2
        expert_input_ids = input_ids[:real_batch_size]
        amateur_input_ids = self.amateur_input_ids_transform(input_ids[real_batch_size:])
        
        if attention_mask is not None:
            assert attention_mask.size(0) == 2 * real_batch_size
            expert_attention_mask = attention_mask[:real_batch_size]
            amateur_attention_mask = attention_mask[real_batch_size:]
        else:
            expert_attention_mask = None
            amateur_attention_mask = None

        if position_ids is not None:
            assert position_ids.size(0) == 2 * real_batch_size
            expert_position_ids = position_ids[:real_batch_size]
            amateur_position_ids = position_ids[real_batch_size:]
        else:
            expert_position_ids = None
            amateur_position_ids = None

        if past_key_values is not None:
            # expert_past_key_values = past_key_values[:-1]
            # amateur_past_key_values = past_key_values[-2]
            # past_input_ids = past_key_values[-1]
            expert_past_key_values, amateur_past_key_values, past_input_ids = past_key_values
            full_expert_input_ids = torch.cat((past_input_ids, expert_input_ids), dim=-1)
        else:
            expert_past_key_values = None
            amateur_past_key_values = None
            full_expert_input_ids = expert_input_ids

        assert inputs_embeds is None
        assert labels is None

        assert not output_attentions
        assert not output_hidden_states
        assert return_dict

        output_expert: CausalLMOutputWithPast = self.expert(
            input_ids=expert_input_ids,
            attention_mask=expert_attention_mask,
            position_ids=expert_position_ids,
            past_key_values=expert_past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **({} if cache_position is None else {'cache_position': cache_position}),
        )

        handles = []
        if self.amateur_mask_layer is not None:
            mask_start, mask_end = self.amateur_mask_layer
            for layer_idx in range(mask_start, mask_end):
                handle = self.amateur.model.layers[layer_idx].register_forward_hook(
                    batch_skip_decoder_layer_hook, with_kwargs=True
                )
                handles.append(handle)

        output_amateur: CausalLMOutputWithPast = self.amateur(
            input_ids=amateur_input_ids.to(self.amateur.device),
            attention_mask=amateur_attention_mask.to(self.amateur.device) \
                if attention_mask is not None else None,
            position_ids=amateur_position_ids.to(self.amateur.device) \
                if position_ids is not None else None,
            past_key_values=amateur_past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **({} if cache_position is None else {'cache_position': cache_position}),
        )

        for handle in handles:
            handle.remove()

        expert_logits = output_expert.logits
        amateur_logits = output_amateur.logits.to(expert_logits.device)

        vocab_size = min(expert_logits.shape[-1], amateur_logits.shape[-1])
        if vocab_size < expert_logits.shape[-1] or vocab_size < amateur_logits.shape[-1]:
            warnings.warn(
                'The two model has different vocab '
                f'({expert_logits.shape[-1]} vs {amateur_logits.shape[-1]}). '
                'Using the smaller one.'
            )
            expert_logits = expert_logits[..., :vocab_size]
            amateur_logits = amateur_logits[..., :vocab_size]

        if self.use_ensemble:
            cd_logits = torch.log(
                (torch.softmax(expert_logits, dim=-1) + torch.softmax(amateur_logits, dim=-1)) / 2
            )
        else:
            cutoff = math.log(self.alpha) + expert_logits.max(dim=-1, keepdim=True).values
            if self.dynamic_beta_scale:
                expert_probs = torch.softmax(expert_logits, dim=-1)
                entropy = torch.sum(-expert_probs * torch.log(expert_probs), dim=-1, keepdim=True)
                beta = entropy * self.beta
            else:
                beta = self.beta
            diffs = (1 + beta) * expert_logits - beta * amateur_logits
            cd_logits = diffs.masked_fill(expert_logits < cutoff, -float('inf'))

        if use_cache:
            # past_key_values = (
            #     output_expert.past_key_values + (output_amateur.past_key_values, full_expert_input_ids)
            # )
            past_key_values = output_expert.past_key_values, output_amateur.past_key_values, full_expert_input_ids
        else:
            past_key_values = None

        cd_logits = torch.where(
            self.cd_condition(full_expert_input_ids, expert_logits[:, -1])[:, None, None],
            cd_logits,
            expert_logits
        )

        cd_logits = torch.cat([cd_logits, cd_logits], dim=0)

        outputs = CausalLMOutputWithPast(
            logits=cd_logits,
            past_key_values=past_key_values,
        )
        outputs['expert_logits'] = expert_logits
        outputs['amateur_logits'] = amateur_logits
        return outputs
    
    def prepare_inputs_for_generation(
        self,
        *args,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            expert_past_key_values, amateur_past_key_values, past_input_ids = past_key_values
        else:
            expert_past_key_values, amateur_past_key_values = None, None
            
        inputs = self.expert.prepare_inputs_for_generation(
            *args, past_key_values=expert_past_key_values, **kwargs
        )
        
        if past_key_values is not None:
            amateur_inputs = self.expert.prepare_inputs_for_generation(
                *args, past_key_values=amateur_past_key_values, **kwargs
            )
            inputs['past_key_values'] = (inputs['past_key_values'], amateur_inputs['past_key_values'], past_input_ids)
            
        return inputs
        
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return (
            LlamaForCausalLM._reorder_cache(past_key_values[:-1], beam_idx),
            LlamaForCausalLM._reorder_cache(past_key_values[-1], beam_idx),
        )



T = TypeVar('T')
def load_hf_model(
    path: str, 
    bit: Literal[4, 8, None, 32] = 4, 
    autoload_cls: Optional[Type[T]] = None,
    use_flash_attention_2: bool = False,
    cpu_offload: bool = False,
    attn_implementation: Optional[str] = None,
) -> Union[T, PreTrainedModel]:
    assert bit in (4, 8, None)
    kwargs = {}
    
    if path == 'baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints':
        kwargs['revision'] = 'train_00220B'
    
    if bit == 4:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
        )
    elif bit == 8:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    if bit == 32 or (bit is None and os.environ.get('USE_FP32_MODEL', False)):
        torch_dtype = torch.float32
        print('Loading model in fp32!')
    else:
        torch_dtype = 'auto'

    device_map = os.environ.get('DEVICE_MAP', 'auto')
    if device_map == 'none':
        device_map = None

    use_flash_attention_2 = (
        device_map is not None and
        torch.cuda.get_device_capability()[0] >= 8 and
        torch_dtype != torch.float32 and
        not os.environ.get('DISABLE_FLASH_ATTN', False) and
        use_flash_attention_2
    )

    if autoload_cls is None:
        autoload_cls = AutoModelForCausalLM
    
    print(f'Loading model {path}')
    print(f'{device_map = }')
    
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # Workaround for Baichuan 2 13B
    if hasattr(config, 'gradient_checkpointing'):
        config.gradient_checkpointing = False
    
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation

    model = autoload_cls.from_pretrained(
        path,
        config=config,
        # device_map='balanced_low_0',
        # device_map='auto',
        device_map=device_map,
        max_memory={'cpu': '256GiB', 0: '4GiB'} if cpu_offload else None,
        # max_memory={
        #     0: '2GiB',
        #     1: '8GiB',
        #     2: '8GiB',
        # },
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attention_2=use_flash_attention_2,
        trust_remote_code=True,
        **kwargs,
    )
    return model


def load_tokenizer(path: str) -> PreTrainedTokenizer:
    use_fast = bool(os.environ.get('FAST_TOK', False))
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=use_fast, legacy=False, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


class CastToFP32(nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dtype in (torch.bfloat16, torch.float16):
            return X.to(dtype=torch.float32)
        else:
            return X


def make_model_cast_to_fp32(module: nn.Module, enable_warmup=True) -> None:
    if not os.environ.get('ENABLE_FP32', None):
        return
    
    if getattr(module, '_modified_has_cast_to_fp32', False):
        warnings.warn('module already has cast to fp32, returning')
        return
    
    # Workaround for baichuan 7b
    if enable_warmup and isinstance(module, PreTrainedModel):
        with torch.no_grad():
            module(input_ids=torch.ones(1, 1, dtype=torch.long).to(device=module.device))

    module._modified_has_cast_to_fp32 = True

    for submodule in module.children():
        make_model_cast_to_fp32(submodule, enable_warmup=False)
    names = [
        name
        for name, p in chain(
            module.named_parameters(recurse=False), 
            module.named_buffers(recurse=False)
        )
    ]
    for name in names:
        parametrize.register_parametrization(module, name, CastToFP32(), unsafe=True)
    
