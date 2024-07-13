from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, MistralForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP, LlamaAttention,
)
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from typing import Iterable, Dict, List, Literal, Optional, Sequence, Type, Union, Tuple, TypeVar
import math
import random

from .stats import stats


T = TypeVar('T', MistralForCausalLM, LlamaForCausalLM)


def get_skip_cd_cls(cls: Type[T]) -> Type[T]:
    class SkipLayerCD(cls):
        _is_test_contrastive_model = True

        def __init__(self, config):
            super().__init__(config)
            self.alpha = 0.1
            self.beta = 0.5
            self.enable_cd = True
            self.share_kv_cache = False
            self.random_mask_range = None
            self.random_mask_layer_num = 1
            self.amateur_temperature = 1.0
            self.cd_type: Literal['normal', 'top2'] = 'normal'

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
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            if not self.enable_cd:
                return super().forward(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    past_key_values=past_key_values, 
                    inputs_embeds=inputs_embeds, 
                    labels=labels, 
                    use_cache=use_cache, 
                    output_attentions=output_attentions, 
                    output_hidden_states=output_hidden_states, 
                    return_dict=return_dict,
                    **({} if cache_position is None else {'cache_position': cache_position}),
                )

            assert return_dict is not False
            assert input_ids is not None and inputs_embeds is None
            # assert input_ids.shape[0] == 1
            assert labels is None

            if self.random_mask_range is not None:
                start_idx, end_idx = self.random_mask_range
                mask_layer_idx = random.randrange(start_idx, end_idx)
                mask_layer(self, mask_layer_idx, mask_layer_idx + self.random_mask_layer_num)

            original_batch_size = input_ids.shape[0]

            input_ids = torch.cat((input_ids, input_ids), dim=0)
            attention_mask = torch.cat((attention_mask, attention_mask), dim=0) \
                if attention_mask is not None else None
            position_ids = torch.cat((position_ids, position_ids), dim=0) \
                if position_ids is not None else None

            outputs: CausalLMOutputWithPast = super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                past_key_values=past_key_values, 
                inputs_embeds=inputs_embeds, 
                labels=labels, 
                use_cache=use_cache, 
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                return_dict=True,
                **({} if cache_position is None else {'cache_position': cache_position}),
            )

            if self.share_kv_cache and outputs.past_key_values is not None:
                for key_state, value_state in outputs.past_key_values:
                    assert key_state.shape[0] == value_state.shape[0] == 2 * original_batch_size
                    key_state[original_batch_size:, :] = key_state[:original_batch_size, :]
                    value_state[original_batch_size:, :] = value_state[:original_batch_size, :]
                del key_state, value_state

            expert_logits = outputs.logits[:original_batch_size]
            amateur_logits = outputs.logits[original_batch_size:] / self.amateur_temperature
            
            if self.cd_type == 'normal':
                cutoff = math.log(self.alpha) + expert_logits.max(dim=-1, keepdim=True).values
                
                diffs = expert_logits - amateur_logits
                diffs *= self.beta
                diffs += expert_logits
                diffs.masked_fill_(expert_logits < cutoff, -float('inf'))
                cd_logits = diffs
                # diffs = (1 + self.beta) * expert_logits - self.beta * amateur_logits
                # cd_logits = diffs.masked_fill(expert_logits < cutoff, -float('inf'))
            else:
                assert 0, self.cd_type

            output = CausalLMOutputWithPast(
                logits=cd_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            output['expert_logits'] = expert_logits.clone()
            output['amateur_logits'] = amateur_logits.clone()
            return output
    
    return SkipLayerCD


Test: Type = get_skip_cd_cls(LlamaForCausalLM)


def batch_skip_decoder_layer_hook(
    layer: LlamaDecoderLayer, 
    args: tuple[Tensor], 
    kwargs: dict[str, Optional[Tensor]], 
    output: tuple[Tensor, ...],
) -> tuple[Tensor]:
    assert len(args) == 1
    input_hidden_states = args[0]
    output_hidden_states = output[0]
    assert input_hidden_states.shape == output_hidden_states.shape
    
    return input_hidden_states, *output[1:]


def half_batch_skip_decoder_layer_hook(
    layer: LlamaDecoderLayer, 
    args: tuple[Tensor], 
    kwargs: dict[str, Optional[Tensor]], 
    output: tuple[Tensor, ...],
) -> tuple[Tensor]:
    assert len(args) == 1
    input_hidden_states = args[0]
    output_hidden_states = output[0]
    assert input_hidden_states.shape == output_hidden_states.shape
    # assert input_hidden_states.shape[0] == output_hidden_states.shape[0] == 2
    assert len(input_hidden_states.shape) == len(output_hidden_states.shape) == 3
    original_batch_size = input_hidden_states.shape[0] // 2
    
    output_hidden_states[original_batch_size:, :, :] = input_hidden_states[original_batch_size:, :, :]
    
    return output_hidden_states, *output[1:]


def half_batch_last_token_skip_decoder_layer_hook(
    layer: LlamaDecoderLayer, 
    args: tuple[Tensor], 
    kwargs: dict[str, Optional[Tensor]], 
    output: tuple[Tensor, ...],
) -> tuple[Tensor]:
    assert len(args) == 1
    input_hidden_states = args[0]
    output_hidden_states = output[0]
    assert input_hidden_states.shape == output_hidden_states.shape
    # assert input_hidden_states.shape[0] == output_hidden_states.shape[0] == 2
    assert len(input_hidden_states.shape) == len(output_hidden_states.shape) == 3
    original_batch_size = input_hidden_states.shape[0] // 2
    
    output_hidden_states[original_batch_size:, -1:, :] = input_hidden_states[original_batch_size:, -1:, :]
    
    return output_hidden_states, *output[1:]


def half_batch_before_last_token_skip_decoder_layer_hook(
    layer: LlamaDecoderLayer, 
    args: tuple[Tensor], 
    kwargs: dict[str, Optional[Tensor]], 
    output: tuple[Tensor, ...],
) -> tuple[Tensor]:
    assert len(args) == 1
    input_hidden_states = args[0]
    output_hidden_states = output[0]
    assert input_hidden_states.shape == output_hidden_states.shape
    # assert input_hidden_states.shape[0] == output_hidden_states.shape[0] == 2
    assert len(input_hidden_states.shape) == len(output_hidden_states.shape) == 3
    original_batch_size = input_hidden_states.shape[0] // 2
    
    output_hidden_states[original_batch_size:, :-1, :] = input_hidden_states[original_batch_size:, :-1, :]
    
    return output_hidden_states, *output[1:]


def mask_layer(model: Test, start: int, end: int, only_last: Union[bool, str] = False):
    assert model._is_test_contrastive_model

    if not hasattr(model, 'mask_hooks'):
        model.mask_hooks = []

    hooks: list[RemovableHandle] = model.mask_hooks
    for handle in hooks:
        handle.remove()
    hooks.clear()
    
    if hasattr(model, 'transformer'): # Bloom
        layers = model.transformer.h
    elif hasattr(model, 'model'): # Others
        layers = model.model.layers
    else:
        assert 0

    for idx in range(start, min(end, len(layers))):
        layer = layers[idx]
            
        if only_last == 'prefill':
            handle = layer.register_forward_hook(half_batch_before_last_token_skip_decoder_layer_hook, with_kwargs=True)            
        elif only_last:
            handle = layer.register_forward_hook(half_batch_last_token_skip_decoder_layer_hook, with_kwargs=True)            
        else:
            handle = layer.register_forward_hook(half_batch_skip_decoder_layer_hook, with_kwargs=True)
        hooks.append(handle)


def mask_layer_continuous(
    model: Test, 
    start: int, end: int, 
    only_last: Union[bool, str] = False,
    skip_mask: Optional[Tensor] = None,
) -> None:
    assert model._is_test_contrastive_model

    if not hasattr(model, 'mask_hooks'):
        model.mask_hooks = []

    hooks: list[RemovableHandle] = model.mask_hooks
    for handle in hooks:
        handle.remove()
    hooks.clear()
    
    tunnel = []
    
    
    def half_batch_skip_decoder_layer_pre_hook_start(
        layer: LlamaDecoderLayer, 
        args: tuple[Tensor],
    ) -> tuple[Tensor]:
        assert len(args) == 1
        input_hidden_states = args[0]
        assert len(input_hidden_states.shape) == 3
        original_batch_size = input_hidden_states.shape[0] // 2
        assert len(tunnel) == 0
        
        tunnel.append(input_hidden_states[original_batch_size:, :, :])
        
        return args
    
    
    def half_batch_skip_decoder_layer_pre_hook_end(
        layer: LlamaDecoderLayer, 
        args: tuple[Tensor],
    ) -> tuple[Tensor]:
        assert len(args) == 1
        input_hidden_states = args[0]
        assert len(input_hidden_states.shape) == 3
        original_batch_size = input_hidden_states.shape[0] // 2
        assert len(tunnel) == 1
        
        amateur_hidden_states: Tensor = tunnel.pop()
        
        if only_last == 'prefill':
            prefill = True
            generation = False
        elif only_last is True:
            assert skip_mask is None
            prefill = False
            generation = True
        elif only_last is False:
            prefill = True
            generation = True
        else:
            assert 0
        
        if prefill:
            if skip_mask is None:
                input_hidden_states[original_batch_size:, :-1, :] = amateur_hidden_states[:, :-1, :]
            else:
                if amateur_hidden_states.size(1) > 1:
                    assert skip_mask.shape == amateur_hidden_states[:, :-1, :].shape[:2]
                    input_hidden_states[original_batch_size:, :-1, :][skip_mask] = \
                        amateur_hidden_states[:, :-1, :][skip_mask]
        
        if generation:
            input_hidden_states[original_batch_size:, -1:, :] = amateur_hidden_states[:, -1:, :]
        
        return args
    
    if start < end:
        handle = model.model.layers[start].register_forward_pre_hook(half_batch_skip_decoder_layer_pre_hook_start)
        hooks.append(handle)
        handle = model.model.layers[end].register_forward_pre_hook(half_batch_skip_decoder_layer_pre_hook_end)
        hooks.append(handle)



def get_dola_hook(
    start: int, end: int, 
    alpha: float = 0.1, beta: float = -1, 
    output_max_layer_ids: bool = False,
    compute_amateur_entropy: bool = False,
    varient: Literal['none', 'm1', 'norm'] = 'none',
):
    assert beta == -1
    assert output_max_layer_ids == False
    if varient == 'm1':
        assert compute_amateur_entropy

    def dola_hook(
        model: LlamaForCausalLM, 
        args: tuple, 
        kwargs: dict, 
        output: CausalLMOutputWithPast,
    ) -> CausalLMOutputWithPast:
        assert kwargs.get('output_hidden_states', model.config.output_hidden_states), kwargs['output_hidden_states']
        assert kwargs.get('return_dict', model.config.return_dict)
        assert isinstance(output, CausalLMOutputWithPast)
        assert output.hidden_states is not None

        logits = output.logits[:, -1:, :]
        assert logits.shape[0] == 1
        hidden_states = torch.stack(
            [
                t[:, -1:, :].to(logits.device) 
                for t in output.hidden_states[start:end]
            ],
            dim=0,
        )
        probs = torch.softmax(logits.float(), dim=-1)

        if varient == 'norm':
            hidden_states = model.model.norm(hidden_states)
        layer_logits = model.lm_head(hidden_states).to(logits.device) # L x B x T(=1) x V
        js_div = torch.zeros(layer_logits.shape[:-1], device=layer_logits.device) # L x B x T(=1)
        for layer_id in range(layer_logits.size(0)):
            layer_probs = torch.softmax(layer_logits[layer_id].float(), dim=-1)
            mean_probs = 0.5 * (probs + layer_probs) 
            js_div[layer_id] = 0.5 * (
                F.kl_div(probs.log()      , mean_probs, reduction='none', log_target=False).mean(dim=-1) + 
                F.kl_div(layer_probs.log(), mean_probs, reduction='none', log_target=False).mean(dim=-1)
            )
        js_div = js_div[:, :, 0].mean(dim=-1) # L
        max_layer_id = torch.argmax(js_div, dim=0) # 1
        # print(max_layer_id.cpu().tolist())
        # if output_max_layer_ids:
        #     output.hidden_states = max_layer_id.cpu() + start
        max_layer_logits = layer_logits[max_layer_id]
        # max_layer_id = max_layer_id[:, :, None, None].expand(-1, -1, -1, layer_logits.size(-1))
        # max_layer_logits = torch.gather(layer_logits.permute(1, 2, 0, 3), dim=2, index=max_layer_id).squeeze(2)
 
        expert_logits = logits
        amateur_logits = max_layer_logits
        assert expert_logits.shape == amateur_logits.shape # B x T(=1) x V

        expert_logits = torch.log_softmax(expert_logits, dim=-1)
        amateur_logits = torch.log_softmax(amateur_logits, dim=-1)
        # if beta == -1:
        #     expert_logits = torch.log_softmax(expert_logits, dim=-1)
        #     amateur_logits = torch.log_softmax(amateur_logits, dim=-1)
        
        if compute_amateur_entropy:
            expert_entropy = -torch.sum(expert_logits * torch.exp(expert_logits), dim=-1).item()
            amateur_entropy = -torch.sum(amateur_logits * torch.exp(amateur_logits), dim=-1).item()
            stats['expert_entropy'].append(expert_entropy)
            stats['amateur_entropy'].append(amateur_entropy)
            
            if varient == 'm1':
                if amateur_entropy < 10.0:
                    print(f'\n{amateur_entropy = }, {amateur_logits.argmax(dim=-1).item()}, {expert_logits.argmax(dim=-1).item()}')
                    amateur_logits = torch.log_softmax(torch.zeros_like(amateur_logits), dim=-1)

        cutoff = math.log(alpha) + expert_logits.max(dim=-1, keepdim=True).values
        diffs = expert_logits - amateur_logits
        # if beta == -1:
        #     diffs = expert_logits - amateur_logits
        # else:
        #     diffs = (1 + beta) * expert_logits - beta * amateur_logits
        cd_logits = diffs.masked_fill(expert_logits < cutoff, -float('inf'))
        
        assert output.logits[:, -1:].shape == cd_logits.shape, (output.logits[:, -1:].shape, cd_logits.shape)
        assert output.logits.dtype == cd_logits.dtype
        output.logits[:, -1:, :] = cd_logits
        output['amateur_logits'] = amateur_logits
        return output
    
    return dola_hook

    