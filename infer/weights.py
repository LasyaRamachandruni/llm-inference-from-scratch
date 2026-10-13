"""Load trained GPT-2 weights from Hugging Face into our from-scratch model.

Only this file depends on `transformers` (install with `pip install -e ".[hf]"`).
The one quirk: Hugging Face's GPT-2 stores its linear layers as `Conv1D` with
weights shaped [in, out], while nn.Linear expects [out, in], so those are transposed.
"""

from __future__ import annotations

from typing import Any

import torch

from .model import GPT, GPTConfig

_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def from_hf_model(hf_model: Any) -> GPT:
    c = hf_model.config
    cfg = GPTConfig(
        vocab_size=c.vocab_size,
        n_positions=c.n_positions,
        n_embd=c.n_embd,
        n_layer=c.n_layer,
        n_head=c.n_head,
        layer_norm_eps=c.layer_norm_epsilon,
    )
    model = GPT(cfg)
    ours = model.state_dict()
    theirs = hf_model.state_dict()

    loaded = set()
    for key, value in theirs.items():
        if key == "lm_head.weight" or key.endswith((".attn.bias", ".attn.masked_bias")):
            continue  # tied to wte / causal-mask buffers we don't need
        name = key.removeprefix("transformer.")
        if name not in ours:
            raise KeyError(f"unexpected weight in checkpoint: {key}")
        if name.endswith(_TRANSPOSED):
            value = value.t()
        if value.shape != ours[name].shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(value.shape)} vs {tuple(ours[name].shape)}")
        with torch.no_grad():
            ours[name].copy_(value)
        loaded.add(name)

    missing = set(ours) - loaded
    if missing:
        raise KeyError(f"weights missing from checkpoint: {sorted(missing)}")
    return model.eval()


def from_pretrained(name: str = "gpt2") -> GPT:
    """Download a GPT-2 checkpoint (gpt2, gpt2-medium, distilgpt2, ...) and load it."""
    from transformers import GPT2LMHeadModel

    return from_hf_model(GPT2LMHeadModel.from_pretrained(name))


def load_tokenizer(name: str = "gpt2") -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)
