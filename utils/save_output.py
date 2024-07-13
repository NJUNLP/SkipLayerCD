import os
import __main__
import inspect
import json
from typing import Union
from .stats import *

main_source = inspect.getsource(__main__)

base_path = './'

def save_json(data: Union[list, dict], name: str, split: str = 'eval'):
    output_dir = os.path.join(base_path, split, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    # open(output_path + '.py', 'w', encoding='utf-8').write(main_source)
    with open(output_path + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)
    if len(stats) > 0:
        with open(output_path + '.stats', 'w', encoding='utf-8') as json_file:
            json.dump(calculate_stats(), json_file, ensure_ascii=False, indent=2)

        