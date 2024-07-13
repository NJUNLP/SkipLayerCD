from typing import Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput


def generate(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    prompt: str,
    max_tokens: int = 4096,
    max_new_tokens: int = 5,
    do_sample: bool = False,
    contain_input: bool = False,
    return_outputs: bool = False,
    add_special_tokens: bool = True,
    skip_special_tokens: bool = True,
    **kwargs,
) -> Union[str, Tuple[str, GenerateOutput]]:
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=add_special_tokens)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    if input_ids.size(1) >= max_tokens:
        print(f'\ninput too long, returning empty string: {input_ids.size(1)} >= {max_tokens}')
        return ''

    outputs: GenerateOutput = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=min(max_new_tokens, max_tokens - input_ids.size(1)),
        do_sample=do_sample,
        return_dict_in_generate=True,
        **kwargs,
    )
    res = outputs.sequences[0]
    # not using this, because tokenizer.decode is only suitable for entire sentence decoding
    # if not contain_input:
    #     res = res[input_ids.size(1):]
    res = tokenizer.decode(res.cpu(), skip_special_tokens=skip_special_tokens)
    if not contain_input:
        res = res[len(tokenizer.decode(input_ids[0], skip_special_tokens=skip_special_tokens)):]
        # res = res[len(prompt):] # not using this to avoid special token in prompt
    if return_outputs:
        return res, outputs
    else:
        return res
