"""Generate text with real GPT-2 weights and compare the decoding methods.

    python -m infer "The meaning of life is" --mode kv
    python -m infer "The meaning of life is" --mode speculative --model gpt2-medium --draft distilgpt2
    python -m infer "Once upon a time" --mode compare
"""

from __future__ import annotations

import argparse
import time

import torch

from .generate import generate, generate_naive
from .sampling import SamplingParams
from .speculative import speculative_generate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--model", default="gpt2", help="gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2")
    ap.add_argument("--draft", default="distilgpt2", help="draft model for speculative mode")
    ap.add_argument("--mode", choices=["naive", "kv", "speculative", "compare"], default="kv")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--k", type=int, default=4, help="draft tokens per round (speculative)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from .weights import from_pretrained, load_tokenizer  # needs `pip install -e ".[hf]"`

    tok = load_tokenizer(args.model)
    model = from_pretrained(args.model)
    params = SamplingParams(args.temperature, args.top_k, args.top_p)
    prompt = tok.encode(args.prompt)

    def run(mode: str):
        gen = torch.Generator().manual_seed(args.seed)
        t0 = time.perf_counter()
        extra = ""
        if mode == "naive":
            out = generate_naive(model, prompt, args.max_new, params, gen)
        elif mode == "kv":
            out = generate(model, prompt, args.max_new, params, gen)
        else:
            draft = run.draft = getattr(run, "draft", None) or from_pretrained(args.draft)
            out, stats = speculative_generate(model, draft, prompt, args.max_new, args.k, params, gen)
            extra = (f", {stats.acceptance_rate:.0%} of drafts accepted, "
                     f"{stats.tokens_per_round:.2f} tokens per {args.model} pass")
        dt = time.perf_counter() - t0
        print(f"\n[{mode}] {len(out) / dt:.1f} tokens/s{extra}")
        print(args.prompt + tok.decode(out))
        return out

    if args.mode == "compare":
        outs = {m: run(m) for m in ("naive", "kv", "speculative")}
        if params.greedy:
            same = outs["naive"] == outs["kv"] == outs["speculative"]
            print(f"\nAll three methods produced identical text: {same}")
    else:
        run(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
