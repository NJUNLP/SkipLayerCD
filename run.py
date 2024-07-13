import os
import random
from utils.load_data import load_data
from utils.model_registry import MODELS
from utils.predict import (
    get_data_type,
    get_mwp_prompt,
    get_mgsm_prompt_base,
    get_aqua_prompt,
    predict_gsm8k,
    predict_mgsm,
    predict_aqua,
)
from utils.setup_models import setup_model_inner
from utils.save_output import save_json

import torch

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def main():
    algorithm = os.environ['ALGO']
    data_name = os.environ['DATA_NAME']
    model_name = os.environ['MODEL']
    random.seed(2023)
    
    output_name = f'{data_name}_{model_name}_{algorithm}'
    print(f'{output_name = }')
    
    testset = load_data(data_name)
    
    data_type = get_data_type(data_name)
    if data_type == 'mwp':
        template = get_mwp_prompt()
    elif data_type == 'aqua':
        template = get_aqua_prompt()
    elif data_type == 'mgsm':
        language = data_name[-2:]
        template = get_mgsm_prompt_base(language)
        
    prefix = '\n\n'.join(template.split('\n\n')[:-1])
        
    tokenizer, model, generate_callback = setup_model_inner(
        algorithm, model_name, prefix=prefix
    )
    
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    if data_type == 'mwp':
        outputs = predict_gsm8k(
            model, MODELS[model_name], tokenizer, testset,
            template, algorithm == 'vanilla', generate_callback,
        )
    elif data_type == 'aqua':
        outputs = predict_aqua(
            model, tokenizer, testset, template,
            algorithm == 'vanilla', generate_callback,
        )
    elif data_type == 'mgsm':
        outputs = predict_mgsm(
            model, tokenizer, testset, template,
            algorithm == 'vanilla', generate_callback,
        )
        
    save_json(outputs, output_name)
        

if __name__ == '__main__':
    main()
