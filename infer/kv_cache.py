"""A preallocated key/value cache.

Without a cache, generating token t means re-running attention over all t
previous tokens, so producing n tokens costs O(n^2) work. The keys and values
of past tokens never change, though, so we can store them once and only compute
the new token's query/key/value at each step.

Layout: [n_layer, batch, n_head, max_len, head_dim] for both K and V.
Memory is allocated up front so decoding never reallocates.

`length` is how many positions are filled. `truncate` just moves that pointer
back, which is how speculative decoding throws away rejected draft tokens.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self,
        n_layer: int,
        batch: int,
        n_head: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        shape = (n_layer, batch, n_head, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.length = 0

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's keys/values for one layer; return all keys/values so far.

        Called once per layer during a forward pass. `length` is only advanced
        (via `advance`) after every layer has written, so all layers see the
        same start position.
        """
        start = self.length
        end = start + k_new.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} positions, have {self.max_len}")
        self.k[layer, :, :, start:end] = k_new
        self.v[layer, :, :, start:end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n: int) -> None:
        self.length += n

    def truncate(self, length: int) -> None:
        """Forget everything after `length` (used to roll back rejected tokens)."""
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot truncate cache of length {self.length} to {length}")
        self.length = length

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2
