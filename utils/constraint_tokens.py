from typing import Sequence
import torch
from torch import LongTensor, FloatTensor
from transformers.generation.logits_process import LogitsProcessor
from transformers import PreTrainedTokenizerBase


class ConstraintTokenProcessor(LogitsProcessor):
    def __init__(self, constraint_ids: Sequence[int]) -> None:
        super().__init__()
        self.constraint_ids = torch.tensor(list(set(constraint_ids)), dtype=torch.long)

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        mask = torch.zeros_like(scores)
        mask[:] = -float('inf')
        mask[..., self.constraint_ids] = 0
        new_scores = scores + mask
        if new_scores.max() > -1e10:
            return new_scores
        else:
            return scores
    

class DisableLineReturnProcessor(LogitsProcessor):
    def __init__(self, line_sep_token: int, prefix: str, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.line_sep_token = line_sep_token
        self.prefix = prefix
        self.tokenizer = tokenizer

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        assert input_ids.size(0) == 1
        input_ids = input_ids.cpu()
        s: str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        line_sep_idx = 0
        while True:
            new_idx = s.find('\n', line_sep_idx+1)
            if new_idx != -1:
                line_sep_idx = new_idx
            else:
                break
        
        prefix_idx = 0
        while True:
            new_idx = s.find(self.prefix, prefix_idx+1)
            if new_idx != -1:
                prefix_idx = new_idx
            else:
                break

        if line_sep_idx >= prefix_idx:
            scores[..., self.line_sep_token] = -float('inf')

        return scores


class ConstraintTokenWithPrefixProcessor(LogitsProcessor):
    active_counter: int = 0

    def __init__(self, constraint_ids: Sequence[int], prefix: str, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.constraint_ids = torch.tensor(constraint_ids, dtype=torch.long)
        self.prefix = prefix
        self.tokenizer = tokenizer

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        assert input_ids.size(0) == 1
        input_ids = input_ids.cpu()

        idx = 0
        while True:
            s = self.tokenizer.decode(input_ids[0, :input_ids.size(1)-idx], skip_special_tokens=True)
            if s.endswith(self.prefix):
                ConstraintTokenWithPrefixProcessor.active_counter += 1
                # print('a')
                mask = torch.zeros_like(scores)
                mask[:] = -float('inf')
                mask[..., self.constraint_ids] = 0
                return scores + mask
            elif input_ids[0, -1-idx] in self.constraint_ids:
                idx += 1
            else:
                return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores
