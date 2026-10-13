"""Load test: continuous vs static batching on the same request trace.

    python -m infer.serving.loadtest offline --requests 64 --repeats 3      # engine in-process, both policies
    python -m infer.serving.loadtest http --url http://localhost:8000 --requests 64 --concurrency 16

Requests are chat prompts (bundled set, or --source oasst1) with requested output
lengths drawn log-uniformly between --min-new and --max-new tokens (a long tail,
like chat replies), all seeded so both policies see exactly the same trace. Every
request generates its full length (EOS is ignored), so timings do not depend on
the weights and random weights give the same numbers as trained ones.

offline: requests arrive as a Poisson process at --rate requests/s (0 = all at
once). Reports output tokens/s, request latency and time to first token
(mean with 95% bootstrap CI over repeats; p50/p90/p99 over requests).

http: drives a running server with --concurrency streaming clients and reports
the same metrics as seen by the clients.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..model import GPT, config_for
from ..prompts import load, load_tokenizer, output_lengths
from ..stats import percentile, summarize
from .engine import Engine, EngineConfig, Request

log = logging.getLogger("infer.loadtest")


@dataclass(frozen=True)
class TraceItem:
    prompt: list[int]
    max_new_tokens: int
    arrival_s: float


def build_trace(
    tokenizer_name: str, n: int, min_new: int, max_new: int, rate: float, source: str, seed: int, max_prompt: int
) -> list[TraceItem]:
    tok = load_tokenizer(tokenizer_name)
    prompts = load(source, seed=seed)
    rng = random.Random(seed)
    texts = [prompts[i % len(prompts)].text for i in range(n)]
    lengths = output_lengths(n, min_new, max_new, seed)
    t, items = 0.0, []
    for text, new in zip(texts, lengths, strict=True):
        if rate > 0:
            t += rng.expovariate(rate)
        items.append(TraceItem(tok.encode(text)[:max_prompt], new, t))
    return items


def run_offline(model: GPT, trace: list[TraceItem], config: EngineConfig) -> dict[str, Any]:
    """Replay the trace in real time against an in-process engine."""
    engine = Engine(model, config)
    pending = sorted(trace, key=lambda r: r.arrival_s)
    requests: list[Request] = []
    t0 = time.perf_counter()
    i = 0
    while i < len(pending) or not engine.idle:
        now = time.perf_counter() - t0
        while i < len(pending) and pending[i].arrival_s <= now:
            item = pending[i]
            req = Request(item.prompt, item.max_new_tokens)
            req.arrival = t0 + item.arrival_s
            requests.append(engine.submit(req))
            i += 1
        if engine.idle and i < len(pending):
            time.sleep(max(0.0, pending[i].arrival_s - (time.perf_counter() - t0)))
            continue
        engine.step()
    wall = time.perf_counter() - t0
    tokens = sum(len(r.output) for r in requests)
    lat = [r.latency for r in requests if r.latency is not None]
    ttft = [r.ttft for r in requests if r.ttft is not None]
    return {
        "wall_s": wall,
        "output_tokens": tokens,
        "tokens_per_s": tokens / wall,
        "requests_per_s": len(requests) / wall,
        "latency_s": {"mean": sum(lat) / len(lat), **{f"p{q}": percentile(lat, q) for q in (50, 90, 99)}},
        "ttft_s": {"mean": sum(ttft) / len(ttft), **{f"p{q}": percentile(ttft, q) for q in (50, 90, 99)}},
        "decode_steps": engine.stats.steps,
        "computed_rows": engine.stats.computed_rows,
        "wasted_row_share": 1 - (tokens - len(requests)) / max(1, engine.stats.computed_rows),
    }


def _hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"{cpu}, {os.cpu_count()} vCPUs, {torch.get_num_threads()} torch threads"


def offline(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.pretrained:
        from ..weights import from_pretrained

        model = from_pretrained(args.model).to(device)
    else:
        torch.manual_seed(0)
        model = GPT(config_for(args.model)).to(device).eval()
    trace = build_trace(
        args.model, args.requests, args.min_new, args.max_new, args.rate, args.source, args.seed, args.max_prompt
    )
    log.info(
        "%d requests, %d output tokens requested, prompts %d-%d tokens",
        len(trace),
        sum(t.max_new_tokens for t in trace),
        min(len(t.prompt) for t in trace),
        max(len(t.prompt) for t in trace),
    )
    runs: dict[str, list[dict[str, Any]]] = {"continuous": [], "static": []}
    for rep in range(args.repeats):
        for policy in ("continuous", "static"):
            cfg = EngineConfig(args.max_slots, args.max_seq_len, args.max_batch_tokens, policy=policy)
            r = run_offline(model, trace, cfg)
            runs[policy].append(r)
            log.info(
                "rep %d %-10s %7.1f tok/s  latency p50 %.2f s  p99 %.2f s  ttft p50 %.2f s",
                rep,
                policy,
                r["tokens_per_s"],
                r["latency_s"]["p50"],
                r["latency_s"]["p99"],
                r["ttft_s"]["p50"],
            )
    summary = {
        policy: {
            "tokens_per_s": summarize([r["tokens_per_s"] for r in rs]),
            "latency_mean_s": summarize([r["latency_s"]["mean"] for r in rs]),
            "latency_p99_s": summarize([r["latency_s"]["p99"] for r in rs]),
            "ttft_mean_s": summarize([r["ttft_s"]["mean"] for r in rs]),
            "wasted_row_share": summarize([r["wasted_row_share"] for r in rs]),
        }
        for policy, rs in runs.items()
    }
    speedup = summary["continuous"]["tokens_per_s"]["mean"] / summary["static"]["tokens_per_s"]["mean"]
    return {
        "command": "python -m infer.serving.loadtest " + " ".join(sys.argv[1:]),
        "hardware": _hardware(),
        "torch": torch.__version__,
        "model": args.model,
        "weights": "trained" if args.pretrained else "random (timing does not depend on weights)",
        "trace": {
            "requests": args.requests,
            "output_tokens": f"log-uniform {args.min_new}-{args.max_new}",
            "arrival_rate_per_s": args.rate or "all at t=0",
            "prompt_source": args.source,
            "seed": args.seed,
        },
        "engine": {
            "max_slots": args.max_slots,
            "max_seq_len": args.max_seq_len,
            "max_batch_tokens": args.max_batch_tokens,
        },
        "summary": summary,
        "throughput_ratio_continuous_over_static": speedup,
        "runs": runs,
    }


async def _http_one(client: Any, url: str, prompt: str, max_tokens: int) -> dict[str, float]:
    t0 = time.perf_counter()
    first = None
    tokens = 0
    body = {"prompt": prompt, "max_tokens": max_tokens, "stream": True, "ignore_eos": True}
    async with client.stream("POST", f"{url}/v1/completions", json=body, timeout=None) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            msg = json.loads(line[6:])
            if "token_id" in msg:
                tokens += 1
                if first is None:
                    first = time.perf_counter() - t0
    return {"latency": time.perf_counter() - t0, "ttft": first or float("nan"), "tokens": tokens}


async def _http(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    prompts = load(args.source, seed=args.seed)
    lengths = output_lengths(args.requests, args.min_new, args.max_new, args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:

        async def one(i: int) -> dict[str, float]:
            async with sem:
                return await _http_one(client, args.url, prompts[i % len(prompts)].text, lengths[i])

        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in range(args.requests)))
        wall = time.perf_counter() - t0
        server = (await client.get(f"{args.url}/metrics")).json()
    lat = [r["latency"] for r in results]
    ttft = [r["ttft"] for r in results]
    tokens = sum(int(r["tokens"]) for r in results)
    return {
        "command": "python -m infer.serving.loadtest " + " ".join(sys.argv[1:]),
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "wall_s": wall,
        "output_tokens": tokens,
        "tokens_per_s": tokens / wall,
        "latency_s": {f"p{q}": percentile(lat, q) for q in (50, 90, 99)},
        "ttft_s": {f"p{q}": percentile(ttft, q) for q in (50, 90, 99)},
        "server": server,
    }


def markdown(result: dict[str, Any]) -> str:
    s = result["summary"]

    def f(x: dict[str, float], d: int = 1) -> str:
        return f"{x['mean']:.{d}f} [{x['ci_low']:.{d}f}, {x['ci_high']:.{d}f}]"

    lines = [
        f"{result['model']} ({result['weights']}), {result['hardware']}, torch {result['torch']}.",
        f"{result['trace']['requests']} requests, outputs {result['trace']['output_tokens']} tokens, "
        f"arrivals: {result['trace']['arrival_rate_per_s']}; {result['engine']['max_slots']} slots. "
        f"Mean [95% CI] over {len(result['runs']['continuous'])} runs.",
        "",
        "| Scheduler | Output tokens/s | Mean latency (s) | p99 latency (s) | Mean TTFT (s) | Wasted rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in ("static", "continuous"):
        p = s[policy]
        lines.append(
            f"| {policy} | {f(p['tokens_per_s'])} | {f(p['latency_mean_s'], 2)} | {f(p['latency_p99_s'], 2)} | "
            f"{f(p['ttft_mean_s'], 2)} | {p['wasted_row_share']['mean']:.0%} |"
        )
    lines.append("")
    lines.append(f"Continuous / static throughput: {result['throughput_ratio_continuous_over_static']:.2f}x.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["offline", "http"])
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-prompt", type=int, default=256, help="truncate prompts to this many tokens")
    ap.add_argument("--rate", type=float, default=0.0, help="Poisson arrivals per second (0 = all at once)")
    ap.add_argument("--source", default="bundled", choices=["bundled", "oasst1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-slots", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-batch-tokens", type=int, default=2048)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None, help="write JSON here (and a .md next to it)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = offline(args) if args.mode == "offline" else asyncio.run(_http(args))
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        if args.mode == "offline":
            args.out.with_suffix(".md").write_text(markdown(result))
        log.info("wrote %s", args.out)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
