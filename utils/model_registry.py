from collections import namedtuple
from typing import Any, Type
import json

ModelDesc = namedtuple(
    'ModelDesc', 
    ['path', 'space_yes_token_id', 'space_no_token_id', 'number_token_ids', 'line_sep_token_ids', 'amateur_model_name'],
    defaults=[None, None, None, None, None, None],
)

MODELS = {
    'llama-3-8b': ModelDesc(
        path='meta-llama/Meta-Llama-3-8B',
        number_token_ids=json.load(open('utils/number_token_ids_llama-3.json')),
        line_sep_token_ids=[1432, 1038, 198, 271],
    ),
    'deepseek-7b': ModelDesc(
        path='deepseek-ai/deepseek-llm-7b-base',
        space_yes_token_id=7587,
        space_no_token_id=2366,
        number_token_ids=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        line_sep_token_ids=[185],
    ),
    'baichuan-2-7b': ModelDesc(
        path='baichuan-inc/Baichuan2-7B-Base',
        number_token_ids=[92335, 92336, 92338, 92354, 92358, 92362, 92369, 92370, 92373, 92383],
        line_sep_token_ids=[5],
        amateur_model_name='baichuan-2-7b-220b',
    ),
    'baichuan-2-7b-220b': ModelDesc(
        path='baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints',
        number_token_ids=[92335, 92336, 92338, 92354, 92358, 92362, 92369, 92370, 92373, 92383],
        line_sep_token_ids=[5],
    ),
    'mistral-7b': ModelDesc(
        path='mistralai/Mistral-7B-v0.1',
        number_token_ids=[28734, 28740, 28750, 28770, 28774, 28781, 28782, 28783, 28784, 28787],
        line_sep_token_ids=[13],
    ),
    'sheared-llama-1b3': ModelDesc(
        path='princeton-nlp/Sheared-LLaMA-1.3B',
        number_token_ids=[29896, 29900, 29906, 29929, 29941, 29945, 29946, 29947, 29953, 29955],
        line_sep_token_ids=[13],
    ),
    'llama-2-13b': ModelDesc(
        path='meta-llama/Llama-2-13b-hf',
        number_token_ids=[29896, 29900, 29906, 29929, 29941, 29945, 29946, 29947, 29953, 29955],
        line_sep_token_ids=[13],
        amateur_model_name='sheared-llama-1b3',
    ),
}


def get_cls(model_name: str) -> Type:
    from transformers import (
        AutoConfig,
        MistralForCausalLM,
        LlamaForCausalLM,
    )
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    MODEL_CLASSES = {
        'mistral-7b': MistralForCausalLM,
        'llama-2-13b': LlamaForCausalLM,
        'llama-3-8b': LlamaForCausalLM,
        'deepseek-7b': LlamaForCausalLM,
    }

    if model_name in MODEL_CLASSES:
        return MODEL_CLASSES[model_name]
    else:
        print('remote model class detected')
        path = MODELS[model_name].path
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        class_ref = config.auto_map['AutoModelForCausalLM']
        model_class = get_class_from_dynamic_module(class_ref, path)
        return model_class

