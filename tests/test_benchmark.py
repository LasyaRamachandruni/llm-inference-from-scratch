"""The benchmark harness runs end to end on tiny models and reports CIs."""

from __future__ import annotations

import torch

from infer import GPT, GPTConfig
from infer.benchmark import bench_batching, bench_decode_step, bench_methods, environment, markdown


def tiny(seed: int, layers: int) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=97, n_positions=128, n_embd=32, n_layer=layers, n_head=4)).eval()


def test_method_comparison_reports_cis_and_acceptance() -> None:
    target, draft = tiny(0, 2), tiny(1, 1)
    prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    out = bench_methods(target, draft, prompts, new_tokens=8, ks=[2], reps=2, naive_prompts=2)
    assert set(out) == {"greedy_no_cache", "kv_cache", "batched", "speculative_k2"}
    spec = out["speculative_k2"]
    assert 0.0 <= spec["acceptance_rate_greedy"]["mean"] <= 1.0
    assert spec["tokens_per_s"]["ci_low"] <= spec["tokens_per_s"]["mean"] <= spec["tokens_per_s"]["ci_high"]
    assert out["kv_cache"]["tokens_per_s"]["n"] == 6  # 3 prompts x 2 repeats
    table = markdown({"env": environment(torch.device("cpu"), True), "methods": out, "speculative": {}})
    assert "| speculative_k2 |" in table and "batched (3 prompts)" in table


def test_cache_and_batching_sections() -> None:
    m = tiny(0, 1)
    rows = bench_decode_step(m, [8, 16], reps=2)
    assert [r["context"] for r in rows] == [8, 16] and rows[0]["kv_cache"]["n"] == 2
    b = bench_batching(m, prompt_len=4, new_tokens=4, batch_sizes=[1, 2], reps=2)
    assert [r["batch"] for r in b] == [1, 2] and all(r["tok_s"] > 0 for r in b)
