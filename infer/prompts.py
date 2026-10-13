"""Prompts and tokenizers for benchmarks, the server and the load test.

Prompt sources:

- bundled (default): 60 chat-style prompts written for this repository
  (infer/data/prompts.jsonl), from one-line questions to paragraph-long requests.
- oasst1: first-turn user prompts from OpenAssistant/oasst1 (Apache-2.0), via the
  `datasets` package (`pip install -e ".[data]"`).

LMSYS-Chat-1M and the ShareGPT dumps were considered and not used: LMSYS-Chat-1M
is gated behind a license agreement that forbids redistribution, and the ShareGPT
dumps are scraped user conversations with no clear license.

Tokenizers: the GPT-2 BPE tokenizer when `transformers` can load it, otherwise a
byte-level fallback (one token per UTF-8 byte), which is enough for timing runs
with random weights where the text itself does not matter.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from importlib import resources
from typing import Protocol

log = logging.getLogger(__name__)


class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """One token per UTF-8 byte (ids 0-255); valid for any GPT-2 sized vocabulary."""

    eos_token_id: int | None = None

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


def load_tokenizer(name: str = "gpt2", allow_fallback: bool = True) -> Tokenizer:
    try:
        from transformers import AutoTokenizer

        tok: Tokenizer = AutoTokenizer.from_pretrained(name)  # type: ignore[assignment]
        return tok
    except Exception as exc:  # no network / no transformers: fall back if allowed
        if not allow_fallback:
            raise
        log.warning("GPT-2 tokenizer unavailable (%s); using the byte-level fallback", type(exc).__name__)
        return ByteTokenizer()


@dataclass(frozen=True)
class Prompt:
    id: int
    text: str


def bundled() -> list[Prompt]:
    raw = resources.files("infer").joinpath("data/prompts.jsonl").read_text(encoding="utf-8")
    return [Prompt(int(r["id"]), str(r["prompt"])) for r in map(json.loads, raw.splitlines()) if r]


def oasst1(n: int, seed: int = 0) -> list[Prompt]:
    """First user turns (English) from OpenAssistant/oasst1, Apache-2.0."""
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    roots = [r["text"] for r in ds if r["parent_id"] is None and r["role"] == "prompter" and r["lang"] == "en"]
    random.Random(seed).shuffle(roots)
    return [Prompt(i, t) for i, t in enumerate(roots[:n])]


def load(source: str = "bundled", n: int | None = None, seed: int = 0) -> list[Prompt]:
    if source == "bundled":
        prompts = bundled()
    elif source == "oasst1":
        prompts = oasst1(n or 500, seed)
    else:
        raise ValueError(f"unknown prompt source '{source}' (bundled, oasst1)")
    return prompts[:n] if n else prompts


def output_lengths(n: int, lo: int = 16, hi: int = 256, seed: int = 0) -> list[int]:
    """Requested output lengths with the long tail of chat replies: log-uniform in [lo, hi]."""
    rng = random.Random(seed)
    return [round(lo * (hi / lo) ** rng.random()) for _ in range(n)]
