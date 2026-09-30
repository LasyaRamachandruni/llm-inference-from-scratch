"""Benchmarks: how much does each technique actually buy?

    python -m infer.benchmark                 # random weights, GPT-2 shapes (no download)
    python -m infer.benchmark --pretrained    # real GPT-2 weights, adds a real speculative run

Speed does not depend on what the weights are, only on their shapes, so the
KV-cache and batching numbers are the same with random or trained weights.
Speculative decoding is different: its speedup depends on how often the draft
model agrees with the target, which needs trained weights. Without
--pretrained we measure the cost ratio between the two models and show the
expected speedup across a range of acceptance rates.

Writes results/benchmarks.json and charts to results/.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .generate import generate, generate_batch, generate_naive
from .model import GPT, config_for
from .speculative import expected_speedup, speculative_generate


def _timeit(fn, reps: int = 5, warmup: int = 1) -> float:
    """Median wall time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _random_model(name: str) -> GPT:
    torch.manual_seed(0)
    return GPT(config_for(name)).eval()


def _load(name: str, pretrained: bool) -> GPT:
    if pretrained:
        from .weights import from_pretrained

        return from_pretrained(name)
    return _random_model(name)


@torch.inference_mode()
def bench_decode_step(model: GPT, contexts: list[int], reps: int) -> list[dict]:
    """Time to produce ONE token when `ctx` tokens came before it.

    Without a cache the model re-reads the whole context; with a cache it only
    processes the new token and reads past keys/values from memory.
    """
    rows = []
    for ctx in contexts:
        ids = torch.randint(0, model.cfg.vocab_size, (1, ctx + 1))

        def no_cache():
            model(ids)

        cache = model.new_cache(1, ctx + 1)
        model(ids[:, :ctx], cache)  # fill the cache once

        def with_cache():
            cache.truncate(ctx)
            model(ids[:, ctx:], cache)

        rows.append(
            {
                "context": ctx,
                "no_cache_ms": _timeit(no_cache, reps) * 1000,
                "kv_cache_ms": _timeit(with_cache, reps) * 1000,
            }
        )
        print(f"  context {ctx:>4}: no cache {rows[-1]['no_cache_ms']:7.1f} ms   "
              f"kv cache {rows[-1]['kv_cache_ms']:6.1f} ms")
    return rows


def bench_end_to_end(model: GPT, prompt_len: int, lengths: list[int]) -> list[dict]:
    """Full generation: tokens/second for naive vs KV-cached decoding."""
    rows = []
    prompt = torch.randint(0, model.cfg.vocab_size, (prompt_len,)).tolist()
    for n in lengths:
        naive = _timeit(lambda: generate_naive(model, prompt, n), reps=1, warmup=0)
        cached = _timeit(lambda: generate(model, prompt, n), reps=3)
        rows.append(
            {
                "new_tokens": n,
                "naive_tok_s": n / naive,
                "kv_tok_s": n / cached,
                "speedup": naive / cached,
            }
        )
        r = rows[-1]
        print(f"  {n:>4} tokens: naive {r['naive_tok_s']:6.1f} tok/s   kv {r['kv_tok_s']:6.1f} tok/s   "
              f"x{r['speedup']:.1f}")
    return rows


def bench_batching(model: GPT, prompt_len: int, new_tokens: int, batch_sizes: list[int]) -> list[dict]:
    """Throughput when decoding several requests together."""
    rows = []
    for b in batch_sizes:
        prompts = [torch.randint(0, model.cfg.vocab_size, (prompt_len,)).tolist() for _ in range(b)]
        t = _timeit(lambda: generate_batch(model, prompts, new_tokens), reps=2)
        rows.append({"batch": b, "tok_s": b * new_tokens / t, "seconds": t})
        print(f"  batch {b:>3}: {rows[-1]['tok_s']:7.1f} tok/s total   ({t:.2f} s per batch)")
    return rows


@torch.inference_mode()
def bench_cost_ratio(target: GPT, draft: GPT, ctx: int, reps: int) -> dict:
    """How expensive is one draft step compared with one target step?"""
    ids = torch.randint(0, target.cfg.vocab_size, (1, ctx + 1))
    out = {}
    for label, m in (("target", target), ("draft", draft)):
        cache = m.new_cache(1, ctx + 1)
        m(ids[:, :ctx], cache)

        def step(m=m, cache=cache):
            cache.truncate(ctx)
            m(ids[:, ctx:], cache)

        out[f"{label}_step_ms"] = _timeit(step, reps) * 1000
    out["cost_ratio"] = out["draft_step_ms"] / out["target_step_ms"]
    print(f"  target step {out['target_step_ms']:.1f} ms, draft step {out['draft_step_ms']:.1f} ms, "
          f"ratio {out['cost_ratio']:.2f}")
    return out


def bench_speculative_real(target: GPT, draft: GPT, prompt: list[int], n: int, ks: list[int]) -> list[dict]:
    """Real speculative decoding with trained weights (needs --pretrained)."""
    base = _timeit(lambda: generate(target, prompt, n), reps=2)
    rows = []
    for k in ks:
        stats = {}

        def run(k=k):
            stats["s"] = speculative_generate(target, draft, prompt, n, k=k)[1]

        t = _timeit(run, reps=2)
        s = stats["s"]
        rows.append(
            {
                "k": k,
                "acceptance_rate": s.acceptance_rate,
                "tokens_per_target_pass": s.tokens_per_round,
                "speedup": base / t,
            }
        )
        print(f"  k={k}: acceptance {s.acceptance_rate:.0%}, {s.tokens_per_round:.2f} tokens/pass, "
              f"speedup x{base / t:.2f}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--target", default="gpt2-medium", help="target model for speculative decoding")
    ap.add_argument("--draft", default="distilgpt2", help="draft model for speculative decoding")
    ap.add_argument("--pretrained", action="store_true", help="download real GPT-2 weights")
    ap.add_argument("--quick", action="store_true", help="fewer, shorter runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reps = 3 if args.quick else 7

    results: dict = {
        "env": {
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "device": "cpu",
            "machine": platform.machine(),
            "pretrained": args.pretrained,
        }
    }

    model = _load(args.model, args.pretrained)
    results["env"]["model"] = args.model
    results["env"]["params_millions"] = round(model.num_params() / 1e6, 1)
    print(f"{args.model}: {results['env']['params_millions']}M parameters, "
          f"{torch.get_num_threads()} CPU threads\n")

    contexts = [32, 128, 256, 512] if args.quick else [32, 64, 128, 256, 384, 512, 768, 1000]
    print("1. Latency of one decode step vs context length")
    results["decode_step"] = bench_decode_step(model, contexts, reps)

    print("\n2. End-to-end generation (16-token prompt)")
    lengths = [32, 128] if args.quick else [32, 64, 128, 256]
    results["end_to_end"] = bench_end_to_end(model, 16, lengths)

    print("\n3. Batched decoding (32-token prompts, 64 new tokens)")
    batches = [1, 4, 16] if args.quick else [1, 2, 4, 8, 16, 32]
    results["batching"] = bench_batching(model, 32, 64, batches)

    print(f"\n4. Speculative decoding: {args.target} (target) + {args.draft} (draft)")
    target = _load(args.target, args.pretrained)
    draft = _load(args.draft, args.pretrained)
    cost = bench_cost_ratio(target, draft, ctx=128, reps=reps)
    alphas = [round(a * 0.05, 2) for a in range(0, 20)]
    ks = [2, 4, 6]
    results["speculative"] = {
        "target": args.target,
        "draft": args.draft,
        **cost,
        "expected": {str(k): [expected_speedup(a, k, cost["cost_ratio"]) for a in alphas] for k in ks},
        "alphas": alphas,
    }
    if args.pretrained:
        from .weights import load_tokenizer

        tok = load_tokenizer(args.target)
        prompt = tok.encode("The history of the printing press begins in")
        results["speculative"]["measured"] = bench_speculative_real(target, draft, prompt, 128, ks)

    (out / "benchmarks.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out / 'benchmarks.json'}")

    try:
        from .plots import make_all

        for p in make_all(results, out):
            print(f"wrote {p}")
    except ImportError:
        print("(install matplotlib to draw charts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
