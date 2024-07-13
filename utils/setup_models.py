from typing import Callable, Literal
from utils.model_registry import MODELS, get_cls
from utils.model_loading import ContrastiveDecodeWithDifferentPromptModel, load_tokenizer
import random
import os

from transformers import BatchEncoding
from utils.model_loading import load_hf_model, make_model_cast_to_fp32
from utils.llama_contrastive_skip_model import get_dola_hook, get_skip_cd_cls, mask_layer
from utils.compute_skip_layer import compute_skip_layer


def setup_model_inner(
    algorithm: Literal['direct', 'vanilla', 'dola', 'sl-h', 'sl-d'],
    model_name: str,
    *,
    prefix: str = None,
):
    model_desc = MODELS[model_name]
    model_dir = model_desc.path
    if model_desc.amateur_model_name is not None:
        amateur_model_dir = MODELS[model_desc.amateur_model_name].path
    else:
        amateur_model_dir = None
    model_cls = get_cls(model_name)
    
    return setup_model(
        algorithm, model_dir,
        prefix=prefix,
        amateur_model_dir=amateur_model_dir,
        model_cls=model_cls,
    )


def setup_model(
    algorithm: Literal['direct', 'vanilla', 'dola', 'sl-h', 'sl-d'],
    # model_name: str,
    model_dir: str,
    *,
    prefix: str = None,
    amateur_model_dir: str = None,
    model_cls: type = None,
):
    tokenizer = load_tokenizer(model_dir)
    generate_callback: Callable[[BatchEncoding], None] = lambda _: None
    
    if algorithm == 'direct':
        model = load_hf_model(model_dir, bit=None)
        make_model_cast_to_fp32(model)
        
    elif algorithm == 'vanilla':
        assert amateur_model_dir is not None
        
        expert = load_hf_model(model_dir, bit=None)
        make_model_cast_to_fp32(expert)
        amateur = load_hf_model(amateur_model_dir, bit=None)
        make_model_cast_to_fp32(amateur)
        model = ContrastiveDecodeWithDifferentPromptModel(
            expert=expert,
            amateur=amateur,
            alpha=0.1,
            beta=0.5,
        )
        
    elif algorithm == 'dola':
        model = load_hf_model(model_dir, bit=None)
        make_model_cast_to_fp32(model)
        model.generation_config.output_hidden_states = True
        model.generation_config.repetition_penalty = 1.2
        end = model.config.num_hidden_layers // 2
        model.register_forward_hook(get_dola_hook(0, end, beta=-1), with_kwargs=True)
        
    elif algorithm == 'sl-h':
        assert model_cls is not None
        model = load_hf_model(model_dir, bit=None, autoload_cls=get_skip_cd_cls(model_cls))
        make_model_cast_to_fp32(model)
        range_end = model.config.num_hidden_layers // 2
        random_mask_layer_num = int(round(model.config.num_hidden_layers / 8))
        
        def generate_callback(_):
            start = random.randrange(4, range_end)
            end = start + random_mask_layer_num
            mask_layer(model, start, end)
        
    elif algorithm == 'sl-d':
        assert model_cls is not None
        model = load_hf_model(model_dir, bit=None, autoload_cls=get_skip_cd_cls(model_cls))
        make_model_cast_to_fp32(model)
        assert prefix is not None
        model.enable_cd = False
        start, end = compute_skip_layer(model, tokenizer, prefix)
        model.enable_cd = True
        mask_layer(model, start, end)
    
    return tokenizer, model, generate_callback
