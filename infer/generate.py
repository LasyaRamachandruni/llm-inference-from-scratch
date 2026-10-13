"""Three ways to generate text, from slowest to fastest.

generate_naive   re-runs the model over the whole sequence for every new token.
generate         prefill once, then decode one token at a time using the KV cache.
generate_batch   several prompts of different lengths at once (left padding + KV cache).

All three produce identical tokens under greedy decoding; the tests check that.
"""

from __future__ import annotations

import torch

from .model import GPT
from .sampling import GREEDY, SamplingParams, next_token


def _tensor(ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.inference_mode()
def generate_naive(
    model: GPT,
    prompt: list[int],
    max_new_tokens: int,
    params: SamplingParams = GREEDY,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Baseline: every step processes the full sequence again. Cost grows with length."""
    device = next(model.parameters()).device
    ids = _tensor(prompt, device)
    window = model.cfg.n_positions
    for _ in range(max_new_tokens):
        logits = model(ids[:, -window:])[:, -1]
        ids = torch.cat([ids, next_token(logits, params, generator).view(1, 1)], dim=1)
    return ids[0, len(prompt) :].tolist()


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: list[int],
    max_new_tokens: int,
    params: SamplingParams = GREEDY,
    generator: torch.Generator | None = None,
) -> list[int]:
    """KV-cached generation: one prefill pass over the prompt, then one token per step."""
    device = next(model.parameters()).device
    cache = model.new_cache(batch=1, max_len=len(prompt) + max_new_tokens)
    logits = model(_tensor(prompt, device), cache)[:, -1]  # prefill
    out: list[int] = []
    for i in range(max_new_tokens):
        tok = next_token(logits, params, generator)
        out.append(int(tok))
        if i < max_new_tokens - 1:
            logits = model(tok.view(1, 1), cache)[:, -1]  # decode: a single new token
    return out


@torch.inference_mode()
def generate_batch(
    model: GPT,
    prompts: list[list[int]],
    max_new_tokens: int,
    params: SamplingParams = GREEDY,
    generator: torch.Generator | None = None,
    pad_id: int = 0,
) -> list[list[int]]:
    """Decode a batch of prompts together.

    Prompts are left-padded to the same length so every row's newest token sits in
    the same column. The pad mask keeps padding out of attention and position ids,
    so each row gets exactly the tokens it would get on its own.
    """
    device = next(model.parameters()).device
    B = len(prompts)
    L = max(len(p) for p in prompts)
    total = L + max_new_tokens

    ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, total), dtype=torch.bool, device=device)
    for row, p in enumerate(prompts):
        ids[row, L - len(p) :] = torch.tensor(p, device=device)
        mask[row, L - len(p) : L] = True

    cache = model.new_cache(batch=B, max_len=total)
    logits = model(ids, cache, mask[:, :L])[:, -1]
    out = torch.empty((B, max_new_tokens), dtype=torch.long, device=device)
    for i in range(max_new_tokens):
        tok = next_token(logits, params, generator)
        out[:, i] = tok
        if i < max_new_tokens - 1:
            mask[:, L + i] = True
            logits = model(tok.view(B, 1), cache, mask[:, : L + i + 1])[:, -1]
    return out.tolist()
