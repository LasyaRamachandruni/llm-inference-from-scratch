"""Turning logits into the next token.

temperature == 0 means greedy decoding. We still return a (one-hot) probability
distribution in that case, which lets speculative decoding use a single
accept/reject rule for both greedy and sampled generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0  # 0 = greedy
    top_k: int = 0  # 0 = off
    top_p: float = 1.0  # 1.0 = off

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


GREEDY = SamplingParams()


def to_probs(logits: torch.Tensor, params: SamplingParams = GREEDY) -> torch.Tensor:
    """Logits [..., V] -> probabilities [..., V] after temperature / top-k / top-p."""
    logits = logits.float()
    if params.greedy:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()

    logits = logits / params.temperature
    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if params.top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        # drop a token if the tokens ranked above it already cover top_p
        drop_sorted = (cum - sorted_logits.softmax(-1)) >= params.top_p
        drop = drop_sorted.scatter(-1, order, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))
    return logits.softmax(-1)


def sample(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one token per row from probabilities [..., V]; returns [...] ids."""
    flat = probs.reshape(-1, probs.shape[-1])
    ids = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    return ids.reshape(probs.shape[:-1])


def next_token(logits: torch.Tensor, params: SamplingParams = GREEDY, generator=None) -> torch.Tensor:
    if params.greedy:
        return logits.argmax(-1)
    return sample(to_probs(logits, params), generator)
