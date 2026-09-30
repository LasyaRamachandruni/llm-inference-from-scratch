"""Speculative decoding (Leviathan et al. 2023; Chen et al. 2023).

Decoding is slow because each token needs a full pass through the big model, and
that pass is limited by reading the weights from memory, not by arithmetic. So
checking several tokens in one pass costs about the same as generating one.

Each round:
  1. A small draft model proposes k tokens, one at a time (cheap).
  2. The big target model scores all k in ONE forward pass.
  3. Draft token i is accepted with probability min(1, p(x)/q(x)), where p and q
     are the target and draft probabilities. At the first rejection we sample a
     replacement from the leftover distribution max(0, p - q), normalized.
  4. If all k are accepted, the target's pass also gives one bonus token for free.

This rule makes the output distribution exactly the target model's, so quality is
unchanged; only speed changes. With greedy decoding p and q are one-hot, the rule
reduces to "accept while the draft matches the target's argmax", and the output
is token-for-token identical to plain greedy decoding with the target.

Both models keep a KV cache. Rejected draft tokens are removed by truncating the
caches, which is just moving a length pointer back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import GPT
from .sampling import GREEDY, SamplingParams, sample, to_probs


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    generated: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        """Tokens produced per target forward pass (plain decoding = 1.0)."""
        return self.generated / self.rounds if self.rounds else 0.0


def _pick(probs: torch.Tensor, params: SamplingParams, generator) -> int:
    return int(probs.argmax()) if params.greedy else int(sample(probs, generator))


@torch.inference_mode()
def speculative_generate(
    target: GPT,
    draft: GPT,
    prompt: list[int],
    max_new_tokens: int,
    k: int = 4,
    params: SamplingParams = GREEDY,
    generator: torch.Generator | None = None,
) -> tuple[list[int], SpecStats]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("target and draft must share a vocabulary")
    device = next(target.parameters()).device

    def t(ids: list[int]) -> torch.Tensor:
        return torch.tensor([ids], dtype=torch.long, device=device)

    seq = list(prompt)
    n0 = len(seq)
    max_len = n0 + max_new_tokens + k + 1
    t_cache = target.new_cache(1, max_len)
    d_cache = draft.new_cache(1, max_len)

    # Invariant: the target cache holds every committed token except the last one,
    # which is fed at the start of the next verification pass.
    if n0 > 1:
        target(t(seq[:-1]), t_cache)
    stats = SpecStats()

    while len(seq) - n0 < max_new_tokens:
        base = len(seq)

        # 1. Draft k tokens. Feed whatever the draft cache hasn't seen yet first.
        d_logits = draft(t(seq[d_cache.length:]), d_cache)[0, -1]
        drafts: list[int] = []
        q_probs: list[torch.Tensor] = []
        for i in range(k):
            q = to_probs(d_logits, params)
            tok = _pick(q, params, generator)
            drafts.append(tok)
            q_probs.append(q)
            if i < k - 1:
                d_logits = draft(t([tok]), d_cache)[0, -1]

        # 2. Verify all k drafts with one target pass over [last, d1, ..., dk].
        #    Row i of the output is the target's distribution for the token after position i.
        p_probs = to_probs(target(t(seq[-1:] + drafts), t_cache)[0], params)  # [k+1, V]

        # 3. Accept / reject.
        n_accepted = 0
        next_tok = None
        for i, tok in enumerate(drafts):
            p, q = p_probs[i], q_probs[i]
            u = torch.rand((), generator=generator, device=p.device)
            if u < torch.clamp(p[tok] / q[tok], max=1.0):
                n_accepted += 1
                continue
            leftover = torch.clamp(p - q, min=0.0)
            next_tok = _pick(leftover / leftover.sum(), params, generator)
            break
        if next_tok is None:  # 4. everything accepted: free bonus token
            next_tok = _pick(p_probs[k], params, generator)

        seq += drafts[:n_accepted] + [next_tok]

        # Roll back caches to drop rejected drafts.
        t_cache.truncate(len(seq) - 1)
        # The draft cache holds seq[:base] + d1..d_{k-1}; keep only the accepted part.
        d_cache.truncate(base + min(n_accepted, k - 1))

        stats.rounds += 1
        stats.drafted += k
        stats.accepted += n_accepted
        stats.generated += n_accepted + 1

    return seq[n0 : n0 + max_new_tokens], stats


def expected_tokens_per_round(alpha: float, k: int) -> float:
    """Expected tokens per target pass if each draft token is accepted with prob alpha."""
    if alpha >= 1.0:
        return k + 1.0
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def expected_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    """Speedup over plain decoding. cost_ratio = draft step time / target step time.

    One round costs k draft steps + 1 target step, and yields
    expected_tokens_per_round tokens instead of 1.
    """
    return expected_tokens_per_round(alpha, k) / (k * cost_ratio + 1)
