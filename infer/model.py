"""GPT-2, written from scratch in plain PyTorch.

Same architecture and parameter names as the original GPT-2, so trained weights
can be loaded (see weights.py). The forward pass optionally takes a KVCache and
a padding mask, which is all the inference tricks in this repo need:

- no cache:           full forward over the whole sequence (baseline)
- cache:              only the new tokens are processed; past K/V come from the cache
- cache + pad_mask:   several left-padded prompts decoded together as one batch
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCache


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# (n_layer, n_head, n_embd) for the public GPT-2 checkpoints.
PRESETS = {
    "distilgpt2": (6, 12, 768),
    "gpt2": (12, 12, 768),
    "gpt2-medium": (24, 16, 1024),
    "gpt2-large": (36, 20, 1280),
    "gpt2-xl": (48, 25, 1600),
}


def config_for(name: str) -> GPTConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown model '{name}', choose from {', '.join(PRESETS)}")
    n_layer, n_head, n_embd = PRESETS[name]
    return GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.layer_idx = layer_idx
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # q, k, v in one matmul
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, cache: KVCache | None, attn_mask: torch.Tensor | None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)  # now covers all past + new positions

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cache, attn_mask):
        x = x + self.attn(self.ln_1(x), cache, attn_mask)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.apply(self._init_weights)
        # GPT-2 scales residual projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def new_cache(self, batch: int, max_len: int) -> KVCache:
        if max_len > self.cfg.n_positions:
            raise ValueError(f"max_len {max_len} exceeds context window {self.cfg.n_positions}")
        p = next(self.parameters())
        return KVCache(
            self.cfg.n_layer, batch, self.cfg.n_head, max_len, self.cfg.head_dim, p.dtype, p.device
        )

    def forward(
        self,
        idx: torch.Tensor,
        cache: KVCache | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits [B, T, vocab] for the T new tokens in `idx`.

        idx:      [B, T] token ids to process now.
        cache:    if given, past keys/values are read from it and the new ones appended.
        pad_mask: [B, past + T] bool, True for real tokens and False for left padding.
                  Covers every position seen so far, not just the new ones.
        """
        B, T = idx.shape
        past = cache.length if cache is not None else 0
        S = past + T  # total sequence length after this step
        device = idx.device

        if pad_mask is not None:
            if pad_mask.shape != (B, S):
                raise ValueError(f"pad_mask must be {(B, S)}, got {tuple(pad_mask.shape)}")
            # Positions count real tokens only, so padding doesn't shift them.
            pos = (pad_mask.long().cumsum(-1) - 1).clamp(min=0)[:, past:]
        else:
            pos = torch.arange(past, S, device=device).unsqueeze(0)

        attn_mask = self._attention_mask(T, past, pad_mask, device)

        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x, cache, attn_mask)
        x = self.ln_f(x)

        if cache is not None:
            cache.advance(T)
        return x @ self.wte.weight.T  # output layer shares weights with the token embedding

    @staticmethod
    def _attention_mask(T: int, past: int, pad_mask, device):
        """Boolean mask [.., T, past+T]; True means 'may attend'."""
        if pad_mask is None and T == 1:
            return None  # a single new token may look at everything before it
        S = past + T
        q_pos = torch.arange(past, S, device=device).unsqueeze(1)  # absolute position of each query
        k_pos = torch.arange(S, device=device).unsqueeze(0)
        causal = k_pos <= q_pos  # [T, S]
        if pad_mask is None:
            return causal
        mask = causal.unsqueeze(0) & pad_mask.unsqueeze(1)  # [B, T, S]
        # A padding query would otherwise have nothing to attend to (NaN). Let every
        # query see itself; padded outputs are never read, and padded keys stay masked.
        mask = mask | (k_pos == q_pos).unsqueeze(0)
        return mask.unsqueeze(1)  # [B, 1, T, S], broadcast over heads
