# Multilingual Contrastive Decoding via Language-agnostic Layers Skipping

## Overview
This repository shares the code and data of our latest work [Multilingual Contrastive Decoding via Language-agnostic Layers Skipping](https://arxiv.org/abs/xxx).

In this work, we find a critical problem of the recent amateur-free contrastive decoding method, DoLa, while working in non-English languages.
Then, we purpose a better amateur-free contrastive decoding approach, Skip Layer, and achieve significant performance improve on both English and Multilingual reasoning benchmarks.

## Usage
Our code is based on the `transformers` library, you should install the dependency to run the code.
```sh
pip install -r requirements.txt
```

We provide a simple integration for the `transformers` library. The LLaMA, Mistral and other similar models should be supported. To load the model with contrastive decoding algorithm, you can use the following code:
```python
from utils.setup_models import setup_model
from transformers import LlamaForCausalLM
model = setup_model(
    algorithm='sl-h', # ['direct', 'vanilla', 'dola', 'sl-h', 'sl-d']
    model_dir='path/to/llama',
    # prefix=..., # need by sl-d
    # amateur_model_dir=..., # need by 'vanilla'
    model_cls=LlamaForCausalLM, # need by 'sl-h' and 'sl-d'
)

# For the "vanilla" algorithm, you should duplicate the input for expert and amateur model:
# model.generate(**tokenizer(['hello'] * 2))

# For other algorithms, just use as a regular llama model
model.generate(**tokenizer(['hello']))
```
You can find more about integration in file `utils/setup_models.py`.

## Experiment
To replicate the experiments conducted in our paper, you can use the following code:
```sh
bash run.sh
```
Note: The result can be different due to the numerical presicion and different evaluation environment. To get more similar result of our paper, you can enable the FP32 inference (see `ENABLE_FP32` in `run.sh`).

The prediction results will be located in `eval/outputs`. The accuracy of the results can be compute by the script `compute_accuracy.py`.

## Citation
If you find this repository helpful, feel free to cite our paper.
```bibtex
stay tuned
```
