"""Benchmarks: how much does each technique actually buy?

    python -m infer.benchmark                          # CPU, random weights, GPT-2 shapes (no download)
    python -m infer.benchmark --pretrained --device cuda --name gpu_t4
                                                       # trained gpt2 + distilgpt2, measured end to end

Speed does not depend on what the weights are, only on their shapes, so the
KV-cache and batching sections give the same timings with random or trained
weights. Speculative decoding is different: its speedup depends on how often
the draft agrees with the target, which needs trained weights. Without
--pretrained the speculative section only measures the draft/target cost ratio
and plugs it into the expected-speedup formula: those numbers are MODELLED, not
measured, and are labelled so in the JSON ("modelled": true).

With --pretrained, every decoding method runs on the same prompts
(infer/data/prompts.jsonl) and the JSON gets, per method: tokens/s and
milliseconds per token (mean with a 95% bootstrap CI over prompts x repeats),
and for speculative decoding the measured acceptance rate (greedy and sampled
at temperature 1) with its CI.

Writes results/<name>.json, results/<name>.md and (with matplotlib) charts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch

from .generate import generate, generate_batch, generate_naive
from .model import GPT, config_for
from .prompts import load as load_prompts
from .prompts import load_tokenizer
from .sampling import SamplingParams
from .speculative import expected_speedup, speculative_generate
from .stats import summarize

log = logging.getLogger("infer.benchmark")


# --- timing -------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_once(fn: Callable[[], object], device: torch.device) -> float:
    """Wall time of one call in seconds, with the GPU queue drained before and after."""
    _sync(device)
    t0 = time.perf_counter()
    fn()
    _sync(device)
    return time.perf_counter() - t0


def time_reps(fn: Callable[[], object], device: torch.device, reps: int, warmup: int = 1) -> list[float]:
    for _ in range(warmup):
        fn()
    return [time_once(fn, device) for _ in range(reps)]


def _ms(samples: Sequence[float]) -> dict[str, float]:
    return summarize([s * 1000 for s in samples])


def _device_of(model: GPT) -> torch.device:
    return next(model.parameters()).device


# --- sections -------------------------------------------------------------------------


@torch.inference_mode()
def bench_decode_step(model: GPT, contexts: list[int], reps: int) -> list[dict[str, Any]]:
    """Time to produce ONE token when `ctx` tokens came before it, without and with the KV cache."""
    device = _device_of(model)
    rows = []
    for ctx in contexts:
        ids = torch.randint(0, model.cfg.vocab_size, (1, ctx + 1), device=device)
        cache = model.new_cache(1, ctx + 1)
        model(ids[:, :ctx], cache)  # fill the cache once

        def no_cache(ids: torch.Tensor = ids) -> None:
            model(ids)

        def with_cache(ids: torch.Tensor = ids, cache: Any = cache, ctx: int = ctx) -> None:
            cache.truncate(ctx)
            model(ids[:, ctx:], cache)

        nc, kv = _ms(time_reps(no_cache, device, reps)), _ms(time_reps(with_cache, device, reps))
        rows.append(
            {"context": ctx, "no_cache_ms": nc["mean"], "kv_cache_ms": kv["mean"], "no_cache": nc, "kv_cache": kv}
        )
        log.info("context %4d: no cache %7.1f ms   kv cache %6.1f ms", ctx, nc["mean"], kv["mean"])
    return rows


def bench_end_to_end(model: GPT, prompt_len: int, lengths: list[int], reps: int) -> list[dict[str, Any]]:
    """Full generation from one random prompt: tokens/s for naive vs KV-cached decoding."""
    device = _device_of(model)
    rows = []
    prompt = torch.randint(0, model.cfg.vocab_size, (prompt_len,)).tolist()
    for n in lengths:
        naive = time_reps(partial(generate_naive, model, prompt, n), device, reps=max(1, reps // 3), warmup=0)
        cached = time_reps(partial(generate, model, prompt, n), device, reps=reps)
        naive_tps, kv_tps = summarize([n / t for t in naive]), summarize([n / t for t in cached])
        rows.append(
            {
                "new_tokens": n,
                "naive_tok_s": naive_tps["mean"],
                "kv_tok_s": kv_tps["mean"],
                "speedup": kv_tps["mean"] / naive_tps["mean"],
                "naive": naive_tps,
                "kv": kv_tps,
            }
        )
        log.info("%4d tokens: naive %6.1f tok/s   kv %6.1f tok/s", n, naive_tps["mean"], kv_tps["mean"])
    return rows


def bench_batching(model: GPT, prompt_len: int, new_tokens: int, batch_sizes: list[int], reps: int) -> list[dict]:
    """Throughput when decoding several requests together (static batch, same lengths)."""
    device = _device_of(model)
    rows = []
    for b in batch_sizes:
        prompts = [torch.randint(0, model.cfg.vocab_size, (prompt_len,)).tolist() for _ in range(b)]
        times = time_reps(partial(generate_batch, model, prompts, new_tokens), device, reps)
        tps = summarize([b * new_tokens / t for t in times])
        rows.append({"batch": b, "tok_s": tps["mean"], "seconds": sum(times) / len(times), "tok_s_ci": tps})
        log.info("batch %3d: %7.1f tok/s total", b, tps["mean"])
    return rows


@torch.inference_mode()
def bench_cost_ratio(target: GPT, draft: GPT, ctx: int, reps: int) -> dict[str, float]:
    """How expensive is one draft step compared with one target step?"""
    device = _device_of(target)
    ids = torch.randint(0, target.cfg.vocab_size, (1, ctx + 1), device=device)
    out: dict[str, float] = {}
    for label, m in (("target", target), ("draft", draft)):
        cache = m.new_cache(1, ctx + 1)
        m(ids[:, :ctx], cache)

        def step(m: GPT = m, cache: Any = cache) -> None:
            cache.truncate(ctx)
            m(ids[:, ctx:], cache)

        out[f"{label}_step_ms"] = _ms(time_reps(step, device, reps))["mean"]
    out["cost_ratio"] = out["draft_step_ms"] / out["target_step_ms"]
    log.info(
        "target step %.2f ms, draft step %.2f ms, ratio %.2f",
        out["target_step_ms"],
        out["draft_step_ms"],
        out["cost_ratio"],
    )
    return out


def bench_methods(
    target: GPT,
    draft: GPT,
    prompts: list[list[int]],
    new_tokens: int,
    ks: list[int],
    reps: int,
    naive_prompts: int,
) -> dict[str, Any]:
    """Every decoding method on the same prompts with trained weights.

    Per prompt and repeat: tokens/s and ms/token. Batched decoding runs all prompts as
    one batch, so it reports aggregate tokens/s and ms per decoding step.
    """
    device = _device_of(target)
    out: dict[str, Any] = {}

    def per_prompt(name: str, fn: Callable[[list[int]], object], subset: list[list[int]], r: int) -> None:
        times = [t for p in subset for t in time_reps(partial(fn, p), device, r)]
        out[name] = {
            "tokens_per_s": summarize([new_tokens / t for t in times]),
            "ms_per_token": summarize([t * 1000 / new_tokens for t in times]),
            "prompts": len(subset),
            "repeats": r,
        }
        log.info(
            "%-16s %7.1f tok/s  %6.2f ms/token",
            name,
            out[name]["tokens_per_s"]["mean"],
            out[name]["ms_per_token"]["mean"],
        )

    per_prompt("greedy_no_cache", lambda p: generate_naive(target, p, new_tokens), prompts[:naive_prompts], 1)
    per_prompt("kv_cache", lambda p: generate(target, p, new_tokens), prompts, reps)

    times = time_reps(lambda: generate_batch(target, prompts, new_tokens), device, reps)
    out["batched"] = {
        "batch": len(prompts),
        "tokens_per_s": summarize([len(prompts) * new_tokens / t for t in times]),
        "ms_per_step": summarize([t * 1000 / new_tokens for t in times]),
        "repeats": reps,
    }
    log.info("batched (%d)     %7.1f tok/s total", len(prompts), out["batched"]["tokens_per_s"]["mean"])

    sampled = SamplingParams(temperature=1.0)
    for k in ks:
        acc_greedy: list[float] = []
        acc_sampled: list[float] = []
        tok_per_pass: list[float] = []
        times = []
        for i, p in enumerate(prompts):
            for _ in range(reps):
                res: dict[str, Any] = {}

                def run(p: list[int] = p, k: int = k, res: dict[str, Any] = res) -> None:
                    res["stats"] = speculative_generate(target, draft, p, new_tokens, k=k)[1]

                times.append(time_once(run, device))
            acc_greedy.append(res["stats"].acceptance_rate)
            tok_per_pass.append(res["stats"].tokens_per_round)
            gen = torch.Generator(device=device).manual_seed(i)
            _, s = speculative_generate(target, draft, p, new_tokens, k=k, params=sampled, generator=gen)
            acc_sampled.append(s.acceptance_rate)
        name = f"speculative_k{k}"
        out[name] = {
            "k": k,
            "tokens_per_s": summarize([new_tokens / t for t in times]),
            "ms_per_token": summarize([t * 1000 / new_tokens for t in times]),
            "acceptance_rate_greedy": summarize(acc_greedy),
            "acceptance_rate_sampled_t1": summarize(acc_sampled),
            "tokens_per_target_pass": summarize(tok_per_pass),
            "speedup_vs_kv_cache": summarize([new_tokens / t for t in times])["mean"]
            / out["kv_cache"]["tokens_per_s"]["mean"],
            "prompts": len(prompts),
            "repeats": reps,
        }
        log.info(
            "%-16s %7.1f tok/s  acceptance %.0f%% (greedy), %.0f%% (T=1)",
            name,
            out[name]["tokens_per_s"]["mean"],
            100 * out[name]["acceptance_rate_greedy"]["mean"],
            100 * out[name]["acceptance_rate_sampled_t1"]["mean"],
        )
    return out


# --- environment and report ------------------------------------------------------------------


def environment(device: torch.device, pretrained: bool) -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    env: dict[str, Any] = {
        "torch": torch.__version__,
        "device": device.type,
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "threads": torch.get_num_threads(),
        "machine": platform.machine(),
        "pretrained": pretrained,
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["cuda"] = torch.version.cuda
    return env


def _fmt(s: dict[str, float], digits: int = 1) -> str:
    return f"{s['mean']:.{digits}f} [{s['ci_low']:.{digits}f}, {s['ci_high']:.{digits}f}]"


def markdown(results: dict[str, Any]) -> str:
    env = results["env"]
    hw = env.get("gpu") or f"{env['cpu']}, {env['threads']} threads"
    lines = [f"Hardware: {hw}; torch {env['torch']}. Means with 95% bootstrap CIs.", ""]
    methods = results.get("methods")
    if methods:
        lines += [
            "| Method | Tokens/s | ms/token | Acceptance (greedy) | Acceptance (T=1) |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, m in methods.items():
            if name == "batched":
                continue
            acc_g = _fmt(m["acceptance_rate_greedy"], 2) if "acceptance_rate_greedy" in m else ""
            acc_s = _fmt(m["acceptance_rate_sampled_t1"], 2) if "acceptance_rate_sampled_t1" in m else ""
            lines.append(f"| {name} | {_fmt(m['tokens_per_s'])} | {_fmt(m['ms_per_token'], 2)} | {acc_g} | {acc_s} |")
        b = methods["batched"]
        lines.append(
            f"| batched ({b['batch']} prompts) | {_fmt(b['tokens_per_s'])} (total) | "
            f"{_fmt(b['ms_per_step'], 2)} per step | | |"
        )
        lines.append("")
    spec = results.get("speculative", {})
    if spec.get("modelled"):
        lines.append(
            f"Speculative expected-speedup curves are modelled from the measured cost ratio "
            f"({spec['cost_ratio']:.2f}), not measured."
        )
    return "\n".join(lines) + "\n"


# --- main --------------------------------------------------------------------------------------


def _load(name: str, pretrained: bool, device: torch.device) -> GPT:
    if pretrained:
        from .weights import from_pretrained

        model = from_pretrained(name)
    else:
        torch.manual_seed(0)
        model = GPT(config_for(name)).eval()
    return model.to(device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2", help="model for the cache and batching sections")
    ap.add_argument("--target", default="gpt2", help="target model for speculative decoding")
    ap.add_argument("--draft", default="distilgpt2", help="draft model for speculative decoding")
    ap.add_argument("--pretrained", action="store_true", help="download trained GPT-2 weights")
    ap.add_argument("--device", default="auto", help="cpu, cuda or auto")
    ap.add_argument("--quick", action="store_true", help="fewer, shorter runs")
    ap.add_argument("--prompts", type=int, default=32, help="prompts for the method comparison")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--reps", type=int, default=0, help="repeats per measurement (0 = 7, or 3 with --quick)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="benchmarks", help="output file stem")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reps = args.reps or (3 if args.quick else 7)
    torch.manual_seed(0)

    results: dict[str, Any] = {
        "env": environment(device, args.pretrained),
        "command": "python -m infer.benchmark " + " ".join(argv if argv is not None else sys.argv[1:]),
    }
    model = _load(args.model, args.pretrained, device)
    results["env"]["model"] = args.model
    results["env"]["params_millions"] = round(model.num_params() / 1e6, 1)
    log.info("%s: %.1fM parameters on %s\n", args.model, model.num_params() / 1e6, device)

    contexts = [32, 128, 256, 512] if args.quick else [32, 64, 128, 256, 384, 512, 768, 1000]
    log.info("1. Latency of one decode step vs context length")
    results["decode_step"] = bench_decode_step(model, contexts, reps)

    log.info("\n2. End-to-end generation (16-token prompt)")
    results["end_to_end"] = bench_end_to_end(model, 16, [32, 128] if args.quick else [32, 64, 128, 256], reps)

    log.info("\n3. Batched decoding (32-token prompts, 64 new tokens)")
    batches = [1, 4, 16] if args.quick else [1, 2, 4, 8, 16, 32]
    results["batching"] = bench_batching(model, 32, 64, batches, max(2, reps // 2))

    log.info("\n4. Speculative decoding: %s (target) + %s (draft)", args.target, args.draft)
    target = model if args.target == args.model else _load(args.target, args.pretrained, device)
    draft = _load(args.draft, args.pretrained, device)
    cost = bench_cost_ratio(target, draft, ctx=128, reps=reps)
    alphas = [round(a * 0.05, 2) for a in range(0, 20)]
    results["speculative"] = {
        "target": args.target,
        "draft": args.draft,
        **cost,
        "modelled": True,
        "note": "expected = speedup formula from the measured cost ratio, not a measurement",
        "expected": {str(k): [expected_speedup(a, k, cost["cost_ratio"]) for a in alphas] for k in args.ks},
        "alphas": alphas,
    }

    if args.pretrained:
        log.info("\n5. All methods on %d chat prompts, %d new tokens, trained weights", args.prompts, args.new_tokens)
        tok = load_tokenizer(args.target, allow_fallback=False)
        prompts = [tok.encode(p.text) for p in load_prompts("bundled", n=args.prompts)]
        methods = bench_methods(
            target, draft, prompts, args.new_tokens, args.ks, reps=max(1, reps // 3), naive_prompts=min(8, len(prompts))
        )
        results["methods"] = methods
        results["speculative"]["measured"] = [
            {"k": m["k"], "acceptance_rate": m["acceptance_rate_greedy"]["mean"], "speedup": m["speedup_vs_kv_cache"]}
            for name, m in methods.items()
            if name.startswith("speculative_k")
        ]

    (out / f"{args.name}.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / f"{args.name}.md").write_text(markdown(results))
    log.info("\nwrote %s and %s", out / f"{args.name}.json", out / f"{args.name}.md")
    try:
        from .plots import make_all

        for p in make_all(results, out, prefix="" if args.name == "benchmarks" else f"{args.name}_"):
            log.info("wrote %s", p)
    except ImportError:
        log.info("(install matplotlib to draw charts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
