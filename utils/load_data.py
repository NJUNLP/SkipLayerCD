import re
import json
import os
import datasets

dir_name = 'data/'

def load_data(data_name: str) -> list[dict[str, object]]:
    if data_name == 'gsm8k':
        return load_gsm8k_socratic(dir_name+'grade-school-math/grade_school_math/data/test_socratic.jsonl')
    elif data_name == 'gsm8k-train':
        return load_gsm8k_socratic(dir_name+'grade-school-math/grade_school_math/data/train_socratic.jsonl')
    elif data_name == 'gsm8k-train100-500':
        return load_gsm8k_socratic(dir_name+'grade-school-math/grade_school_math/data/train_socratic.jsonl')[100:500]
    elif data_name == 'gsm-plus-digits':
        return load_gsm_plus_digits()
    elif data_name == 'aqua':
        return load_aqua(dir_name+'AQuA/dev.json')
    elif data_name == 'aqua-test':
        return load_aqua(dir_name+'AQuA/test.json')
    elif data_name.startswith('mgsm-') and len(data_name) == 7:
        language = data_name[-2:]
        return load_mgsm(dir_name+f'mgsm/mgsm_{language}.tsv')
    else:
        raise RuntimeError(f'Unknown data name: {data_name}')

def load_gsm8k_socratic(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        path = dir_name + path
    data = [json.loads(line.strip()) for line in open(path, 'r')]
    for example in data:
        lines = example['answer'].splitlines()
        answer_num = int(lines[-1][5:].replace(',', '_'))
        subquestions = []
        subanswers = []
        for line in lines[:-1]:
            subquestion, subanswer = line.split(' ** ', maxsplit=1)
            subanswer = re.sub(r'<<.+?>>', '', subanswer)
            subquestions.append(subquestion)
            subanswers.append(subanswer)
        example['answer'] = answer_num
        example['decomposition'] = subquestions
        example['subanswers'] = subanswers
        example['facts'] = []
    return data


def load_gsm_plus_digits() -> list[dict[str, object]]:
    data = datasets.load_from_disk('qintongli_GSM-Plus')['test']
    v = []
    for example in data:
        if not all(ch in '0123456789' for ch in example['answer']):
            continue
        example['subanswers'] = [example['solution']]
        example['answer'] = int(example['answer'])
        v.append(example)
    return v


def load_mgsm(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        path = dir_name + path
    data = [line.strip().split('\t') for line in open(path, 'r')]
    v = []
    for question, answer in data:
        example = {}
        answer_num = int(answer.replace(',', '_'))
        example['question'] = question
        example['answer'] = answer_num
        v.append(example)
    return v


def load_aqua(path: str):
    if not os.path.exists(path):
        path = dir_name + path
    data = [json.loads(line.strip()) for line in open(path, 'r', encoding='utf-8')]
    v = []
    for example in data:
        # question = f"{example['question']} Answer Choices:"
        # for option in example['options']:
        #     letter, content = option.split(')', maxsplit=1)
        #     question += f' ({letter.lower()}) {content}'
        question = example['question']
        options = []
        for option in example['options']:
            letter, content = option.split(')', maxsplit=1)
            letter = letter.lower()
            options.append((letter, content))
        v.append({
            'question': question,
            'options': options,
            'answer': example['correct'].lower(),
        })
    return v


def load_json(path: str):
    if not os.path.exists(path):
        path = dir_name + path
    return json.load(open(path))

