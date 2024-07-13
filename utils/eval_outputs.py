import json
from typing import Optional
import re
import os

from .load_data import load_mgsm, load_data

testset_mgsm  = load_mgsm('data/mgsm/mgsm_en.tsv')

def extract_last_num(text: str) -> Optional[float]:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return None
    
    
def check_gsm8k_or_aqua(ans: str, ref) -> bool:
    return ans.strip() == str(ref)
    

def check_mgsm(cot: str, ans, ref) -> Optional[bool]:
    try:
        if type(ans) is str:
            ans = ans.replace(',', '_')
        ans = float(ans)
    except (ValueError, TypeError):
        ans = extract_last_num(cot)

    if ans is not None:
        return abs(ans - ref) < 1e-3
    else:
        return None


def eval_gsm8k_or_aqua_outputs(data_name: str, path: str, verbose: bool = False) -> float:
    testset = load_data(data_name)
    
    first_k = os.environ.get('FIRST_K', None)
    if first_k is not None:
        first_k = int(first_k)
        print(f'Evaluating {first_k = } samples...')
        testset = testset[:first_k]
    
    outputs = json.load(open(path))
    assert len(outputs) == len(testset)
    correct_count = 0
    for idx, (example, (cot, ans)) in enumerate(zip(testset, outputs)):
        if check_gsm8k_or_aqua(ans, example['answer']):
            correct_count += 1
                
    if verbose:
        print('Target:', path)
        print(f'{len(testset) = }')
        print(f'{correct_count = }')
        print(f'Accuracy = {correct_count / len(testset) * 100:.1f}%')

    return correct_count / len(testset)


def eval_mgsm_outputs(path: str, verbose: bool = False) -> float:
    testset = testset_mgsm
    
    first_k = os.environ.get('FIRST_K', None)
    if first_k is not None:
        first_k = int(first_k)
        print(f'Evaluating {first_k = } samples...')
        testset = testset[:first_k]
    
    outputs = json.load(open(path))
    assert len(outputs) == len(testset)
    correct_count = 0
    number_count = 0
    for idx, (example, (cot, ans)) in enumerate(zip(testset, outputs)):
        correct = check_mgsm(cot, ans, example['answer'])
        if correct is not None:
            number_count += 1
            if correct:
                correct_count += 1
                
    if verbose:
        print('Target:', path)
        print(f'{len(testset) = }')
        print(f'{number_count = }')
        print(f'{correct_count = }')
        print(f'Accuracy = {correct_count / len(testset) * 100:.1f}%')

    return correct_count / len(testset)
