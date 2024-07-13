from calendar import month_name
import json
from typing import Callable, Literal, Optional
from tqdm import tqdm, trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase, BatchEncoding, StoppingCriteriaList, LogitsProcessorList
from transformers.generation.utils import GenerateOutput
import torch
import yaml

from utils.constraint_tokens import ConstraintTokenProcessor
from utils.generate import generate
from utils.model_registry import MODELS, ModelDesc
from .eval_outputs import extract_last_num, check_mgsm, check_gsm8k_or_aqua
from .stop_sequence import StopSequenceCriteria, StopWhenContainSequenceCriteria
from .stats import calculate_stats
import os

from data.mgsm.exemplars import EXEMPLAR_NUMBER_ANSWERS, MGSM_EXEMPLARS, MGSM_START_WORDS

def get_data_type(data_name: str) -> Literal['mwp', 'aqua', 'mgsm']:
    if 'mgsm' in data_name:
        return 'mgsm'
    elif 'gsm' in data_name:
        return 'mwp'
    elif 'aqua' in data_name:
        return 'aqua'
    assert 0, data_name


def get_mwp_prompt(add_generation: bool = True) -> str:
    promptset = yaml.safe_load(open('prompt_mwp.yaml'))
    prompt = ''
    for sample in promptset[:8]:
        prompt += f'Problem: {sample["question"]}\n'
        prompt += f'Solution: {" ".join(sample["subanswers"])}\nFinal Answer: {sample["answer"]}\n'
        prompt += '\n'
        
    if add_generation:
        prompt += 'Problem: {query}\n'
        prompt += 'Solution:'
    
    return prompt


def get_aqua_prompt(add_generation: bool = True) -> str:
    promptset = yaml.safe_load(open('prompt_aqua.yaml'))
    prompt = ''
    for sample in promptset[:8]:
        prompt += f'Problem: {sample["question"]}\n'
        prompt += f'Choices: {sample["options"]}\n'
        prompt += f'Solution: {" ".join(sample["subanswers"])}\nFinal Answer: {sample["answer"]}\n'
        prompt += '\n'
        
    if add_generation:
        prompt += 'Problem: {query}\n'
        prompt += 'Choices: {choices}\n'
        prompt += 'Solution:'
    
    return prompt


def get_mgsm_prompt_base(
    language: Optional[str] = None,
    nshot: Optional[int] = None,
) -> str:
    if language is None:
        language = os.environ.get('MGSM_LANG', 'en')

    assert nshot is None
    if nshot is None:
        assert os.environ.get('NSHOT') is None
        if language == 'te':
            nshot = 2
        elif language in ('th', 'bn'):
            nshot = 4
        else:
            nshot = 8


    promptset = [
        {
            'question': MGSM_EXEMPLARS[language][str(idx+1)]['q'],
            'subanswers': [MGSM_EXEMPLARS[language][str(idx+1)]['a']],
            'answer': EXEMPLAR_NUMBER_ANSWERS[idx],
        }
        for idx in range(len(EXEMPLAR_NUMBER_ANSWERS))
    ]


    prompt_base = ''
    for sample in promptset[:nshot]:
        prompt_base += f'{sample["question"]}\n'
        for subanswer in sample['subanswers']:
            prompt_base += subanswer + '\n'
        prompt_base += '\n'
    prompt_base += MGSM_START_WORDS[language][0] + '{query}\n' + MGSM_START_WORDS[language][1]

    return prompt_base


def predict_aqua(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    testset: list[dict],
    prompt_base: str,
    is_vanilla_cd = False,
    generate_callback: Callable[[BatchEncoding], None] = lambda _: None,
):
    assert tokenizer.padding_side == 'left'
    assert tokenizer.pad_token_id is not None
    assert model.generation_config.pad_token_id is not None
    assert '{query}' in prompt_base

    max_new_tokens = 800
    
    template = get_aqua_prompt(add_generation=True)

    outputs = []
    correct_count = 0
    bar = tqdm(testset)
    for idx, example in enumerate(bar):
        choices = ' '.join(f'({letter}) {content}' for letter, content in example['options'])
        msg = template.format(query=example["question"], choices=choices)
        inputs = tokenizer(
            [msg, msg] if is_vanilla_cd else [msg],
            return_tensors='pt',
            padding="longest",
        )
        generate_callback(inputs)
        output: GenerateOutput = model.generate(
            **inputs.to(device=model.device),
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([StopSequenceCriteria(tokenizer, '\n', strip=False)]),
            return_dict_in_generate=True,
            do_sample=False,
        )
        cot = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        msg, cot = cot, cot[len(msg):]
        msg += 'Final Answer: ('
        res = generate(
            model.expert if is_vanilla_cd else model,
            tokenizer,
            msg, 
            max_new_tokens=1, 
            logits_processor=LogitsProcessorList([
                ConstraintTokenProcessor(
                    tokenizer.convert_tokens_to_ids(list('abcde'))
                )
            ]),
        )
        msg += res + ')'
        ans = res
        if idx == 0:
            print(ans)
            print(msg)
        outputs.append((cot, ans))
        
        if check_gsm8k_or_aqua(ans, example['answer']):
            correct_count += 1
            
        bar.set_postfix({'Acc': correct_count / len(outputs) * 100})

    print(f'{len(testset) = }')
    print(f'{correct_count = }')
    print(f'accuracy = {correct_count / len(testset) * 100:.2f}%')
    print(json.dumps(calculate_stats(contain_data=False), indent=2, ensure_ascii=False))

    return outputs


def predict_gsm8k(
    model: PreTrainedModel,
    model_desc: ModelDesc,
    tokenizer: PreTrainedTokenizerBase,
    testset: list[dict],
    prompt_base: str,
    is_vanilla_cd = False,
    generate_callback: Callable[[BatchEncoding], None] = lambda _: None,
):
    assert tokenizer.padding_side == 'left'
    assert tokenizer.pad_token_id is not None
    assert model.generation_config.pad_token_id is not None
    assert '{query}' in prompt_base

    max_new_tokens = 800
    
    template = get_mwp_prompt(add_generation=True)
    constraint_list = model_desc.number_token_ids + model_desc.line_sep_token_ids + \
        tokenizer.encode('\n\n', add_special_tokens=False)[-1:]

    outputs = []
    correct_count = 0
    bar = tqdm(testset)
    for idx, example in enumerate(bar):
        msg = template.format(query=example["question"])
        inputs = tokenizer(
            [msg, msg] if is_vanilla_cd else [msg],
            return_tensors='pt',
            padding="longest",
        )
        generate_callback(inputs)
        output: GenerateOutput = model.generate(
            **inputs.to(device=model.device),
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([StopSequenceCriteria(tokenizer, '\n', strip=False)]),
            return_dict_in_generate=True,
            do_sample=False,
        )
        cot = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        msg, cot = cot, cot[len(msg):]
        msg += 'Final Answer: '
        res = generate(
            model.expert if is_vanilla_cd else model,
            tokenizer,
            msg, 
            max_new_tokens=20, 
            stopping_criteria=StoppingCriteriaList([StopSequenceCriteria(tokenizer, '\n', strip=False)]),
            logits_processor=LogitsProcessorList([ConstraintTokenProcessor(constraint_list)]),
        )
        msg += res
        ans = res[:-1]
        if idx == 0:
            print(ans)
            print(msg)
        outputs.append((cot, ans))
        if check_gsm8k_or_aqua(ans, example['answer']):
            correct_count += 1
            
        bar.set_postfix({'Acc': correct_count / len(outputs) * 100})

    print(f'{len(testset) = }')
    print(f'{correct_count = }')
    print(f'accuracy = {correct_count / len(testset) * 100:.2f}%')
    print(json.dumps(calculate_stats(contain_data=False), indent=2, ensure_ascii=False))

    return outputs


def predict_mgsm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    testset: list[dict],
    prompt_base: str,
    is_vanilla_cd = False,
    generate_callback: Callable[[BatchEncoding], None] = lambda _: None,
):
    assert tokenizer.padding_side == 'left'
    assert tokenizer.pad_token_id is not None
    assert model.generation_config.pad_token_id is not None
    assert '{query}' in prompt_base


    max_new_tokens = 800
    batch_size = 1

    inputss = []
    for idx in range(0, len(testset), batch_size):
        inputs = tokenizer(
            [
                *[
                    prompt_base.format(query=example["question"])
                    for example in testset[idx:idx+batch_size]
                ],
            ] * (2 if is_vanilla_cd else 1),
            return_tensors='pt',
            padding="longest",
        )
        
        if batch_size > 1 and inputs.input_ids.shape[1] + max_new_tokens >= model.config.max_position_embeddings:
            raise RuntimeError(f'context is not enough long while batch_size > 1')
        inputss.append(inputs)
        

    outputs = []
    correct_count = 0
    number_count = 0
    bar = trange(0, len(testset), batch_size)
    for idx in bar:
        torch.cuda.empty_cache()
        inputs: BatchEncoding = inputss[idx]
        inputss[idx] = None
        
        assert batch_size == 1
        stopping_criteria = StoppingCriteriaList([
            StopWhenContainSequenceCriteria(
                tokenizer, '\n\n', len(tokenizer.decode(inputs.input_ids[0]))
            )
        ])
        
        generate_callback(inputs)
        logits_processor = inputs.pop('logits_processor', None)
        output: GenerateOutput = model.generate(
            **inputs.to(device=model.device),
            # model.config.model_max_length is fix for baichuan 2 13b
            max_new_tokens=min(max_new_tokens, (getattr(model.config, 'max_position_embeddings', None) or model.config.model_max_length) - inputs.input_ids.shape[1]),
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            do_sample=False,
            logits_processor=logits_processor,
        )

        for in_batch_idx in range(len(testset[idx:idx+batch_size])):
            msg = prompt_base.format(query=testset[idx + in_batch_idx]["question"])
            res = tokenizer.decode(
                output.sequences[in_batch_idx].cpu(), 
                skip_special_tokens=True,
            )[len(tokenizer.decode(inputs.input_ids[in_batch_idx], skip_special_tokens=True)):]
            msg += res
            ans = extract_last_num(res)

            correct = check_mgsm(res, ans, testset[idx + in_batch_idx]['answer'])

            if correct is not None:
                number_count += 1
                if correct:
                    correct_count += 1

            if idx + in_batch_idx == 0 or os.environ.get('STEP_VERBOSE', False):
                print(ans)
                print(msg)

            outputs.append((res, ans))
            
        bar.set_postfix({'Acc': correct_count / len(outputs) * 100})

    print(f'{len(testset) = }')
    print(f'{number_count = }')
    print(f'number_ratio = {number_count / len(testset) * 100:.2f}%')
    print(f'{correct_count = }')
    print(f'accuracy = {correct_count / len(testset) * 100:.2f}%')
    print(json.dumps(calculate_stats(contain_data=False), indent=2, ensure_ascii=False))

    return outputs
