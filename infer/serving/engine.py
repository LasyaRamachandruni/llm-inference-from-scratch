"""Continuous batching: iteration-level scheduling over a slot-based KV cache.

Static batching (what generate_batch does) starts a batch and runs it until its
longest request is done. Short requests sit finished in the batch, still being
computed, and requests that arrive meanwhile wait for the whole batch. Chat
traffic has a long tail of output lengths, so most of that time is wasted.

Continuous batching (Orca, Yu et al. 2022; vLLM) schedules at every decoding
step instead:

  each iteration
    1. admit waiting requests while there is a free slot and the iteration's token
       budget (max_batch_tokens: prompt tokens being prefilled + one token per
       running sequence) allows; each admitted prompt is prefilled into its slot
       and its first token sampled
    2. one decoding step for every running sequence, as a single batched forward
    3. sequences that hit their token limit (or EOS) leave at once, freeing the slot

The KV cache is preallocated per slot: [n_layer, max_slots, n_head, max_seq_len,
head_dim]. Each slot keeps its own length, so sequences of any length share one
forward pass; an attention mask hides each row's unused positions. Running
sequences are kept packed in slots 0..n-1 (a finished sequence's slot is filled by
moving the last one into it), so the attention reads are plain tensor views.

`EngineConfig(policy="static")` turns the same engine into a static batcher:
admit up to max_slots requests only when the engine is empty, and keep finished
rows in the batch (computed, output discarded) until the whole batch is done.
That is the baseline the load test compares against.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch

from ..kv_cache import KVCache
from ..model import GPT
from ..sampling import GREEDY, SamplingParams, next_token

log = logging.getLogger(__name__)

TokenCallback = Callable[["Request", int, bool], None]


@dataclass
class EngineConfig:
    max_slots: int = 16  # sequences decoded together (KV cache rows)
    max_seq_len: int = 1024  # prompt + generated tokens per sequence (KV cache length)
    max_batch_tokens: int = 2048  # tokens processed per iteration: prefill + one per running sequence
    policy: Literal["continuous", "static"] = "continuous"


_ids = itertools.count()


@dataclass
class Request:
    prompt: list[int]
    max_new_tokens: int
    params: SamplingParams = GREEDY
    eos_token_id: int | None = None
    seed: int | None = None
    on_token: TokenCallback | None = None
    id: int = field(default_factory=lambda: next(_ids))
    # filled in by the engine
    output: list[int] = field(default_factory=list)
    arrival: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    finish_reason: str | None = None
    generator: torch.Generator | None = None

    @property
    def done(self) -> bool:
        return self.finish_reason is not None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token_at is None else self.first_token_at - self.arrival

    @property
    def latency(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.arrival


class SlotKVCache:
    """Per-slot KV cache. Implements the `update` hook the model's attention layers call."""

    def __init__(self, model: GPT, slots: int, max_len: int) -> None:
        cfg = model.cfg
        p = next(model.parameters())
        shape = (cfg.n_layer, slots, cfg.n_head, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=p.dtype, device=p.device)
        self.v = torch.zeros(shape, dtype=p.dtype, device=p.device)
        self.lengths = torch.zeros(slots, dtype=torch.long, device=p.device)
        self.max_len = max_len
        self._n = 0  # rows taking part in the current step
        self._span = 0  # positions read in the current step

    def begin_step(self, n: int, span: int) -> None:
        self._n, self._span = n, span

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write one new position per row at that row's own length; return keys/values so far."""
        rows = torch.arange(self._n, device=k_new.device)
        pos = self.lengths[: self._n]
        self.k[layer, rows, :, pos] = k_new[:, :, 0]
        self.v[layer, rows, :, pos] = v_new[:, :, 0]
        return self.k[layer, : self._n, :, : self._span], self.v[layer, : self._n, :, : self._span]

    def load_prefill(self, slot: int, prefill: KVCache, length: int) -> None:
        self.k[:, slot, :, :length] = prefill.k[:, 0, :, :length]
        self.v[:, slot, :, :length] = prefill.v[:, 0, :, :length]
        self.lengths[slot] = length

    def move(self, src: int, dst: int) -> None:
        n = int(self.lengths[src])
        self.k[:, dst, :, :n] = self.k[:, src, :, :n]
        self.v[:, dst, :, :n] = self.v[:, src, :, :n]
        self.lengths[dst] = n


@dataclass
class EngineStats:
    steps: int = 0
    prefills: int = 0
    generated_tokens: int = 0
    computed_rows: int = 0  # sequence-steps computed, including finished rows kept by static batching
    completed: int = 0


class Engine:
    def __init__(self, model: GPT, config: EngineConfig | None = None) -> None:
        self.model = model.eval()
        self.config = config or EngineConfig()
        if self.config.max_seq_len > model.cfg.n_positions:
            raise ValueError(
                f"max_seq_len {self.config.max_seq_len} exceeds the context window {model.cfg.n_positions}"
            )
        self.cache = SlotKVCache(model, self.config.max_slots, self.config.max_seq_len)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # running[i] lives in cache slot i
        self.last_tokens: list[int] = []  # newest token of each running sequence, not yet in the cache
        self.stats = EngineStats()
        self.device = next(model.parameters()).device

    # --- public API ---------------------------------------------------------------------

    def submit(self, req: Request) -> Request:
        self.validate(req)
        self.waiting.append(req)
        return req

    def validate(self, req: Request) -> None:
        """Raise ValueError if the request can never run; also seeds its sampler.

        Only reads the config, so the server can call it from any thread.
        """
        total = len(req.prompt) + req.max_new_tokens
        if not req.prompt:
            raise ValueError("empty prompt")
        if req.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if total > self.config.max_seq_len:
            raise ValueError(f"prompt + max_new_tokens = {total} exceeds max_seq_len {self.config.max_seq_len}")
        if len(req.prompt) + 1 > self.config.max_batch_tokens:
            raise ValueError(f"prompt of {len(req.prompt)} tokens exceeds max_batch_tokens")
        if req.seed is not None and not req.params.greedy:
            req.generator = torch.Generator(device=self.device).manual_seed(req.seed)

    @property
    def idle(self) -> bool:
        return not self.waiting and not self.running

    def step(self) -> list[Request]:
        """One scheduler iteration. Returns the requests that finished in it."""
        finished = self._admit()
        self._evict()  # requests that finished on their first token
        if self.running:
            finished += self._decode()
        return finished

    def fail(self, req: Request) -> None:
        """End a request with an error and tell its listener."""
        req.finish_reason = "error"
        req.finished_at = time.perf_counter()
        if req.on_token is not None:
            req.on_token(req, -1, True)

    def run_until_done(self) -> None:
        while not self.idle:
            self.step()

    # --- scheduling ----------------------------------------------------------------------

    def _admit(self) -> list[Request]:
        cfg = self.config
        if cfg.policy == "static" and self.running:
            return []
        budget = cfg.max_batch_tokens - len(self.running)
        finished: list[Request] = []
        longest_prompt, longest_output = 0, 0
        while self.waiting and len(self.running) < cfg.max_slots:
            nxt = self.waiting[0]
            need = len(nxt.prompt)
            if need > budget:
                break
            if cfg.policy == "static":
                # finished rows keep growing until the batch ends: the longest prompt plus the
                # longest output must fit every row
                lp, lo = max(longest_prompt, need), max(longest_output, nxt.max_new_tokens)
                if lp + lo > cfg.max_seq_len:
                    break
                longest_prompt, longest_output = lp, lo
            req = self.waiting.popleft()
            budget -= need
            try:
                finished += self._prefill(req)
            except Exception:  # one bad request must not take the batch down
                log.exception("prefill failed for request %d", req.id)
                if self.running and self.running[-1] is req:
                    self.running.pop()
                    self.last_tokens.pop()
                self.fail(req)
                finished.append(req)
        return finished

    @torch.inference_mode()
    def _prefill(self, req: Request) -> list[Request]:
        slot = len(self.running)
        n = len(req.prompt)
        kv = self.model.new_cache(1, n)
        logits = self.model(torch.tensor([req.prompt], device=self.device), kv)[0, -1]
        self.cache.load_prefill(slot, kv, n)
        self.running.append(req)
        self.last_tokens.append(-1)
        self.stats.prefills += 1
        tok = int(next_token(logits, req.params, req.generator))
        return self._emit(slot, tok)

    @torch.inference_mode()
    def _decode(self) -> list[Request]:
        n = len(self.running)
        lengths = self.cache.lengths[:n]
        span = int(lengths.max()) + 1
        if span > self.cache.max_len:
            raise RuntimeError("KV cache overflow; admission should have prevented this")
        tokens = torch.tensor(self.last_tokens, device=self.device).view(n, 1)
        x = self.model.wte(tokens) + self.model.wpe(lengths.view(n, 1))
        mask = (torch.arange(span, device=self.device).view(1, span) <= lengths.view(n, 1)).view(n, 1, 1, span)
        self.cache.begin_step(n, span)
        for block in self.model.h:
            x = block(x, self.cache, mask)
        logits = (self.model.ln_f(x) @ self.model.wte.weight.T)[:, 0]
        self.cache.lengths[:n] += 1
        self.stats.steps += 1
        self.stats.computed_rows += n

        greedy = all(r.params.greedy for r in self.running)
        next_ids = (
            logits.argmax(-1).tolist()
            if greedy
            else [int(next_token(logits[i], r.params, r.generator)) for i, r in enumerate(self.running)]
        )
        finished: list[Request] = []
        for i, tok in enumerate(next_ids):
            if not self.running[i].done:
                finished += self._emit(i, tok)
        self._evict()
        return finished

    def _emit(self, slot: int, tok: int) -> list[Request]:
        req = self.running[slot]
        now = time.perf_counter()
        if req.first_token_at is None:
            req.first_token_at = now
        req.output.append(tok)
        self.last_tokens[slot] = tok
        self.stats.generated_tokens += 1
        if req.eos_token_id is not None and tok == req.eos_token_id:
            req.finish_reason = "eos"
        elif len(req.output) >= req.max_new_tokens:
            req.finish_reason = "length"
        if req.done:
            req.finished_at = now
            self.stats.completed += 1
        if req.on_token is not None:
            req.on_token(req, tok, req.done)
        return [req] if req.done else []

    def _evict(self) -> None:
        """Remove finished sequences, keeping running ones packed in slots 0..n-1."""
        if self.config.policy == "static" and not all(r.done for r in self.running):
            return  # static batching: the batch stays together until its longest request is done
        i = 0
        while i < len(self.running):
            if self.running[i].done:
                last = len(self.running) - 1
                if i != last:
                    self.cache.move(last, i)
                    self.running[i] = self.running[last]
                    self.last_tokens[i] = self.last_tokens[last]
                self.running.pop()
                self.last_tokens.pop()
            else:
                i += 1
        if not self.running:
            self.cache.lengths.zero_()
