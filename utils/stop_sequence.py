import torch
from transformers import PreTrainedTokenizerBase
from transformers.generation.stopping_criteria import StoppingCriteria

class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, tokenizer, s: str, strip: bool = True) -> None:
        super().__init__()
        if strip:
            s = s.strip()
        self.tokenizer = tokenizer
        self.s = s
        self.strip = strip

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # assert input_ids.shape[0] == 1
        input_ids = input_ids[:1]
        res = self.tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=True)[0]
        if self.strip:
            res = res.strip()
        end = res.endswith(self.s)
        # print(end)
        return end


class StopWhenContainSequenceCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, s: str, prefix_len: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.s = s

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        input_ids = input_ids.cpu()
        assert input_ids.shape[0] == 1 or \
            (input_ids.shape[0] == 2 and (input_ids[0] == input_ids[1]).all())
        res = self.tokenizer.batch_decode(input_ids.cpu())[0]
        res = res[self.prefix_len:]
        end = self.s in res
        # print(end)
        return end