import torch
from torch import Tensor
from transformers import BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithPast

def select_layer_based_on_entropy(entropy: Tensor) -> tuple[int, int]:
    # print(entropy)
    num_layers = entropy.size(0)
    entropy_diff = entropy[:-1] - entropy[1:]
    entropy_diff_pooling = (entropy_diff[1:] + entropy_diff[:-1]) / 2
    entropy_is_going_down = (entropy_diff_pooling > 0.1).tolist()
    
    num_skip = int(round(num_layers / 8))
    
    for idx in range(0, num_layers - 2):
        mask_end = idx + 2
        mask_start = mask_end - num_skip
        
        if idx < num_skip + 4:
            continue
        
        if entropy_is_going_down[idx] and (entropy[mask_end] > entropy[mask_end+1:]).all():
            return mask_start, mask_end


def compute_skip_layer(
    model,
    tokenizer,
    prefix: str,
):
    num_total_layers = len(model.model.layers)
    
    inputs: BatchEncoding = tokenizer(prefix, return_tensors='pt')
    input_ids: Tensor = inputs.input_ids.clone()
    with torch.no_grad():
        outputs: CausalLMOutputWithPast = model(
            **inputs.to(model.device),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states

    X = torch.arange(num_total_layers)
    Y = []

    for hidden_state in hidden_states[:-1]:
        hidden_state: Tensor = hidden_state[:, :-1, :]
        
        logits: Tensor = model.lm_head(model.model.norm(hidden_state)).cpu().float()
        lprobs = torch.log_softmax(logits, dim=-1)
        entropy = -(lprobs * torch.exp(lprobs)).sum(dim=-1)
        
        Y.append(entropy.mean().item())
        
    Y = torch.tensor(Y)

    start, end = select_layer_based_on_entropy(Y)
    print(f'selected based on entropy: [{start}, {end})')
    return start, end

