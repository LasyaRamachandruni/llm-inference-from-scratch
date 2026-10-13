"""Correctness tests. Every speedup must give exactly the same answer as the baseline.

Models are tiny and randomly initialized so the suite runs in seconds on a CPU.
float64 is used so that argmax comparisons are not flipped by rounding noise.
"""

import pytest
import torch

from infer import GPT, GPTConfig, SamplingParams, generate, generate_batch, generate_naive
from infer.sampling import to_probs
from infer.speculative import expected_speedup, expected_tokens_per_round, speculative_generate


def tiny(seed=0, vocab=97, layers=2, embd=32, heads=4, sharpen=1.0) -> GPT:
    torch.manual_seed(seed)
    m = GPT(GPTConfig(vocab_size=vocab, n_positions=128, n_embd=embd, n_layer=layers, n_head=heads))
    with torch.no_grad():
        m.wte.weight.mul_(sharpen)  # larger embeddings -> peakier, more distinct distributions
    return m.double().eval()


PROMPT = [5, 17, 3, 42, 8, 11]


# --- KV cache -----------------------------------------------------------------


def test_cached_logits_match_full_forward():
    m = tiny()
    ids = torch.tensor([[*PROMPT, 1, 2, 3, 4]])
    full = m(ids)

    cache = m.new_cache(1, ids.shape[1])
    parts = [m(ids[:, :6], cache)]  # prefill
    for i in range(6, ids.shape[1]):  # decode one at a time
        parts.append(m(ids[:, i : i + 1], cache))
    incremental = torch.cat(parts, dim=1)

    assert torch.allclose(full, incremental, atol=1e-10)
    assert cache.length == ids.shape[1]


def test_cache_truncate_and_overflow():
    m = tiny()
    cache = m.new_cache(1, 8)
    m(torch.tensor([PROMPT]), cache)
    cache.truncate(3)
    assert cache.length == 3
    with pytest.raises(ValueError):
        cache.truncate(5)  # cannot grow by truncating
    with pytest.raises(ValueError):
        m(torch.tensor([[1] * 6]), cache)  # 3 + 6 > 8


def test_cache_larger_than_context_rejected():
    with pytest.raises(ValueError):
        tiny().new_cache(1, 10_000)


# --- generation ---------------------------------------------------------------


def test_kv_generation_matches_naive():
    m = tiny()
    assert generate(m, PROMPT, 20) == generate_naive(m, PROMPT, 20)


def test_sampled_generation_is_reproducible_and_matches_naive():
    m = tiny(sharpen=10)
    params = SamplingParams(temperature=0.8, top_k=20)
    a = generate(m, PROMPT, 15, params, torch.Generator().manual_seed(1))
    b = generate_naive(m, PROMPT, 15, params, torch.Generator().manual_seed(1))
    assert a == b


def test_batch_matches_individual_runs():
    m = tiny()
    prompts = [[1, 2, 3], [9, 8, 7, 6, 5, 4, 3], [42], [10, 20, 30, 40]]
    batched = generate_batch(m, prompts, 12)
    for p, out in zip(prompts, batched, strict=False):
        assert out == generate(m, p, 12), f"mismatch for prompt {p}"


# --- sampling -----------------------------------------------------------------


def test_top_k_and_top_p_filtering():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    p = to_probs(logits, SamplingParams(temperature=1.0, top_k=2))
    assert (p[2:] == 0).all() and torch.isclose(p.sum(), torch.tensor(1.0))

    full = logits.softmax(-1)  # [0.636, 0.234, 0.086, ...]
    p = to_probs(logits, SamplingParams(temperature=1.0, top_p=0.8))
    assert (p[:2] > 0).all() and (p[2:] == 0).all()  # top 2 cover 0.87 >= 0.8
    assert torch.allclose(p[:2], full[:2] / full[:2].sum())


def test_greedy_probs_are_one_hot():
    p = to_probs(torch.tensor([0.1, 2.0, -1.0]))
    assert p.tolist() == [0.0, 1.0, 0.0]


# --- speculative decoding -----------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 5])
def test_speculative_greedy_matches_target_exactly(k):
    target = tiny(seed=0, layers=3)
    draft = tiny(seed=123, layers=1)  # a different, unrelated model
    expected = generate(target, PROMPT, 25)
    out, stats = speculative_generate(target, draft, PROMPT, 25, k=k)
    assert out == expected
    assert stats.generated >= 25


def test_speculative_with_perfect_draft_accepts_everything():
    target = tiny()
    out, stats = speculative_generate(target, target, PROMPT, 20, k=4)
    assert out == generate(target, PROMPT, 20)
    assert stats.acceptance_rate == 1.0
    assert stats.tokens_per_round == 5.0  # k accepted + 1 bonus per target pass


def test_speculative_single_token_prompt():
    target, draft = tiny(), tiny(seed=7, layers=1)
    out, _ = speculative_generate(target, draft, [3], 10, k=3)
    assert out == generate(target, [3], 10)


class Bigram(torch.nn.Module):
    """A stand-in model whose next-token logits depend only on the previous token.

    Lets the test pick target and draft distributions that clearly disagree, and
    compute the exact answer. Implements the same interface speculative decoding uses.
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.table = torch.nn.Parameter(table, requires_grad=False)
        self.cfg = GPTConfig(vocab_size=table.shape[0], n_positions=128, n_layer=1, n_head=1, n_embd=1)

    def new_cache(self, batch, max_len):
        from infer import KVCache

        return KVCache(1, batch, 1, max_len, 1, dtype=self.table.dtype)

    def forward(self, idx, cache=None, pad_mask=None):
        if cache is not None:
            cache.advance(idx.shape[1])
        return self.table[idx]


def test_speculative_sampling_matches_target_distribution():
    """The accept/reject rule must reproduce the target's distribution exactly,
    even when the draft model disagrees with it a lot.

    Checks the joint distribution of the first two generated tokens (36 outcomes)
    against the exact target probabilities.
    """
    V = 6
    g = torch.Generator().manual_seed(0)
    target = Bigram(torch.randn(V, V, generator=g, dtype=torch.float64))
    draft = Bigram(torch.randn(V, V, generator=g, dtype=torch.float64))
    params = SamplingParams(temperature=1.0)
    prompt = [1, 2]

    P = target.table.softmax(-1)  # P[prev, next]
    true = P[2].unsqueeze(1) * P  # joint probability of (token1, token2) after "2"

    counts = torch.zeros(V, V, dtype=torch.float64)
    n, accepted, drafted = 6000, 0, 0
    for _ in range(n):
        (a, b), stats = speculative_generate(target, draft, prompt, 2, k=3, params=params, generator=g)
        counts[a, b] += 1
        accepted += stats.accepted
        drafted += stats.drafted

    assert 0.1 < accepted / drafted < 0.9, "draft should disagree with target often"
    tv = 0.5 * (counts / n - true).abs().sum().item()
    assert tv < 0.05, f"total variation {tv:.3f} too large"


def test_expected_speedup_formula():
    assert expected_tokens_per_round(1.0, 4) == 5.0
    assert expected_tokens_per_round(0.0, 4) == 1.0
    assert abs(expected_tokens_per_round(0.5, 2) - 1.75) < 1e-12  # 1 + 0.5 + 0.25
    assert expected_speedup(0.0, 4, 0.0) == 1.0
    assert expected_speedup(0.8, 4, 0.1) > 2.0


# --- loading real GPT-2 weights ----------------------------------------------


def test_matches_hugging_face_gpt2():
    """Load a (random, offline) Hugging Face GPT-2 into our model; logits must match."""
    transformers = pytest.importorskip("transformers")
    from infer.weights import from_hf_model

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=100, n_positions=64, n_embd=48, n_layer=2, n_head=4)
    hf = transformers.GPT2LMHeadModel(cfg).eval()
    ours = from_hf_model(hf)

    ids = torch.tensor([[1, 5, 9, 33, 70, 2]])
    with torch.no_grad():
        assert torch.allclose(ours(ids), hf(ids).logits, atol=1e-5)
