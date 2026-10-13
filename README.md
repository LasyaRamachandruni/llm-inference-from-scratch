# LLM Inference from Scratch

GPT-2 inference written from the ground up in plain PyTorch, with the techniques that make real LLM serving fast: a KV cache, batched decoding, speculative decoding and a continuous-batching server. Each one is implemented by hand, tested against the slow baseline for identical output, and benchmarked.

| Technique | What it fixes | Result | Status |
|---|---|---|---|
| **KV cache** | Re-computing attention over the whole history for every new token | Per-token time stays flat at **~25 ms** instead of growing to **1.2 s** at 1,000 tokens (**47×**) | measured, CPU |
| **Batched decoding** | One request at a time leaves the hardware idle | **8.2×** more tokens/second at batch size 32 | measured, CPU |
| **Continuous batching** | Static batches wait for their longest request | **1.91× the throughput** of static batching (84.6 vs 44.3 tokens/s) and half the mean latency, same 64-request trace | measured, CPU |
| **Speculative decoding** | Every token needs a full pass through the big model | **1.7× is modelled, not measured**: the speedup formula at an assumed 80% acceptance with the measured draft/target cost ratio (23%, distilgpt2 drafting for gpt2-medium) | T4 measurement pending |

CPU numbers: Intel Xeon 2.8 GHz, 2 vCPUs, 2 torch threads, GPT-2 124M shapes. Timing does not depend on the weight values, so these runs use random weights (no download).

**Pending: measured speculative decoding and GPU numbers.** [`notebooks/gpu_benchmark_colab.ipynb`](notebooks/gpu_benchmark_colab.ipynb) runs on a Colab T4 with trained weights (gpt2 target, distilgpt2 draft) and measures acceptance rate, tokens/s and latency per token for greedy, KV-cache, batched and speculative decoding, with 95% confidence intervals, plus continuous vs static batching on the GPU. It writes `results/gpu_t4.*` and `results/serving_gpu_t4.*`. Until those files are committed, the speculative row above is a model, not a result.

No inference libraries are used: the model, attention masking, caches, batching, the speculative accept/reject rule and the scheduler are all in [`infer/`](infer). Hugging Face is used only to download trained weights and the tokenizer.

## Results

### KV cache: time per token no longer grows with context

![Time per token vs context length](results/decode_latency.png)

Without a cache, generating each token re-runs the model over every previous token, so the cost grows linearly per token (quadratically in total). With the cache, keys and values of past tokens are stored once and only the new token is processed.

End to end, from a 16-token prompt:

| New tokens | Naive (tok/s) | KV cache (tok/s) | Speedup |
|---:|---:|---:|---:|
| 32 | 15.3 | 43.5 | 2.8× |
| 64 | 12.4 | 43.7 | 3.5× |
| 128 | 9.0 | 46.9 | 5.2× |
| 256 | 5.5 | 44.1 | 8.0× |

### Batching: more requests for almost the same cost

![Throughput vs batch size](results/batching.png)

Decoding one token is limited by reading the model's weights from memory, not by arithmetic. Processing 32 requests together reads the weights once for all of them, so total throughput rises from 41 to 337 tokens/second. Prompts of different lengths are left-padded, and a padding mask keeps padding out of both attention and position ids, so each request gets exactly the tokens it would get on its own (tested).

### Continuous batching: requests join and leave at every step

From [`results/serving_cpu.md`](results/serving_cpu.md), produced by
`python -m infer.serving.loadtest offline --model gpt2 --requests 64 --repeats 3 --max-slots 16`:
64 chat prompts arriving at once, requested outputs log-uniform between 16 and 256 tokens (6,438 tokens in total), 16 sequences decoded together, mean and 95% CI over 3 runs.

| Scheduler | Output tokens/s | Mean latency (s) | p99 latency (s) | Mean time to first token (s) | Computed rows wasted |
|---|---:|---:|---:|---:|---:|
| static | 44.3 [44.0, 44.5] | 61.7 [61.6, 61.9] | 140.9 [140.3, 141.8] | 48.5 [48.3, 48.7] | 53% |
| continuous | 84.6 [83.3, 86.3] | 37.8 [36.7, 38.6] | 74.6 [73.1, 75.6] | 22.0 [21.2, 22.6] | 0% |

A static batch runs until its longest request is done: with a long tail of output lengths, 53% of the rows it computed belonged to requests that had already finished, and new requests waited for the whole batch. The continuous scheduler fills a freed slot on the next step, so every computed row produces a token. On a GPU the gap should grow, since a larger batch costs relatively less there; the T4 notebook measures it.

### Speculative decoding: several tokens per big-model pass (modelled)

![Modelled speedup vs acceptance rate](results/speculative_speedup.png)

A small draft model guesses *k* tokens; the target model checks all of them in **one** forward pass. Accepted guesses are free tokens. The accept/reject rule from [Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192) guarantees the output follows the target model's distribution exactly, so speed improves with no change in quality.

These curves are **modelled**: they combine the measured cost of one draft step (23% of a target step for distilgpt2 vs gpt2-medium on this CPU) with the standard formula for expected accepted tokens. They say what the speedup *would be* at each acceptance rate; the acceptance rate itself needs trained weights and is measured by the T4 notebook. Two takeaways from the model:
- Below ~50–60% acceptance it is **slower** than normal decoding; the draft has to agree with the target often enough to pay for itself.
- More draft tokens (larger *k*) only pay off when acceptance is high.

## How it works

```
infer/
  model.py         GPT-2 (attention, MLP, blocks) with optional KV cache and padding mask
  kv_cache.py      preallocated K/V tensors; truncate() rolls back rejected tokens
  generate.py      naive, KV-cached and batched generation
  speculative.py   draft -> verify -> accept/reject -> roll back caches
  sampling.py      greedy, temperature, top-k, top-p
  weights.py       loads Hugging Face GPT-2 checkpoints into the from-scratch model
  serving/
    engine.py      continuous batching: per-slot KV cache, iteration-level scheduler
    server.py      FastAPI server, request queue, streaming (server-sent events)
    loadtest.py    seeded request traces; continuous vs static, or against a running server
  benchmark.py     timing harness with confidence intervals; writes results/<name>.json and .md
  prompts.py       bundled chat prompts, optional oasst1 loader, tokenizer fallback
  stats.py         bootstrap confidence intervals and percentiles
  plots.py         charts in results/
notebooks/         gpu_benchmark_colab.ipynb (T4)
tests/             correctness tests (run offline in seconds)
```

**KV cache.** Each layer writes the new token's keys and values into a preallocated buffer at position `length` and attends over `[0, length]`. `length` only advances after all layers have written, so every layer sees the same position. Preallocating means decoding never reallocates memory.

**Batching with padding.** Prompts are left-padded so every row's newest token sits in the same column. A boolean mask `[batch, seq]` marks real tokens; it is combined with the causal mask, and position ids are computed from the running count of real tokens so padding doesn't shift them. Each query may always attend to itself, which avoids an all-masked row (NaN) on padding positions.

**Speculative decoding.** Per round:
1. The draft generates *k* tokens with its own KV cache.
2. The target scores `[last token, d1 … dk]` in one pass, producing *k + 1* next-token distributions.
3. Draft token *i* is kept with probability `min(1, p(x) / q(x))`. At the first rejection, a replacement is sampled from `max(0, p − q)`, normalized. If all *k* pass, the target's last distribution gives a bonus token.
4. Both caches are truncated back to the accepted prefix.

With greedy decoding `p` and `q` are one-hot, so the same rule becomes "keep drafts while they match the target's top choice", and the output is **token-for-token identical** to normal greedy decoding.

**Continuous batching.** The KV cache has one row ("slot") per running sequence, `[layer, slot, head, max_seq_len, head_dim]`, and each slot keeps its own length. Every scheduler iteration:
1. admits waiting requests while a slot is free and the iteration's token budget allows (`max_batch_tokens` = prompt tokens being prefilled + one token per running sequence); each new prompt is prefilled into its slot and its first token sampled;
2. runs one decoding step for all running sequences as a single batched forward through the model's own blocks: each row writes its new key/value at its own length, and an attention mask hides the positions it has not filled;
3. removes finished sequences immediately, moving the last running sequence into the freed slot so the batch stays packed and attention reads are plain tensor views.

The same engine with `policy="static"` admits a batch only when it is empty and keeps finished rows until the whole batch is done, which is the baseline above.

## Serving

```bash
pip install -e ".[serve,hf]"
INFER_PRETRAINED=1 python -m infer.serving.server        # gpt2 with trained weights on :8000

curl -N localhost:8000/v1/completions -H 'content-type: application/json' \
     -d '{"prompt": "The history of the printing press", "max_tokens": 64, "stream": true}'
curl localhost:8000/metrics                               # queue depth, running, tokens/s

python -m infer.serving.loadtest http --requests 64 --concurrency 16   # load-test a running server
```

Configuration is by environment variable: `INFER_MODEL`, `INFER_PRETRAINED`, `INFER_DEVICE`, `INFER_POLICY` (continuous or static), `INFER_MAX_SLOTS`, `INFER_MAX_SEQ_LEN`, `INFER_MAX_BATCH_TOKENS`, `INFER_MAX_QUEUE` (beyond it the server answers 503). Invalid requests get 422. A Dockerfile builds a CPU image of the server.

Load-test prompts: 60 chat prompts bundled with the repository by default, or first user turns from [OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache-2.0) with `--source oasst1`. LMSYS-Chat-1M was not used because its license forbids redistribution, and the ShareGPT dumps have no clear license.

## Tests

```bash
pip install -e ".[dev]"
pytest              # 35 tests with coverage
ruff check . && ruff format --check . && mypy
```

The tests check correctness, not just that code runs:
- Cached logits equal a full forward pass; cached, naive and batched generation give identical tokens.
- Speculative greedy output equals the target's own greedy output for *k* = 1, 3, 5, even with an unrelated draft model.
- **Speculative sampling reproduces the target's exact distribution:** the joint distribution of two sampled tokens over 6,000 runs matches the true probabilities (total variation < 0.05), using a draft that disagrees with the target most of the time.
- **Continuous batching changes nothing about the output:** 25 requests of different lengths, arriving while others are decoding, each get exactly the tokens `generate()` gives them alone, under both scheduling policies.
- The server streams every token, validates input and keeps serving when one request fails.
- Our model loaded with Hugging Face GPT-2 weights produces the same logits as Hugging Face's own implementation.

CI (GitHub Actions) runs ruff, mypy and the tests on CPU torch, starts the server and streams a completion through it, and builds the Docker image. Dependencies are locked in `requirements/` (Python 3.12).

## Try it

```bash
pip install -e ".[hf,plots]"

# generate with real GPT-2 weights
python -m infer "The history of the printing press" --mode kv
python -m infer "The history of the printing press" --mode speculative --model gpt2-medium --draft distilgpt2

# run naive, KV cache and speculative on the same prompt and confirm identical output
python -m infer "Once upon a time" --mode compare --model gpt2-medium

# benchmarks: random weights on CPU (no download), or trained weights on a GPU
python -m infer.benchmark
python -m infer.benchmark --pretrained --device cuda --name gpu_t4
```

The CPU KV-cache and batching numbers come from `python -m infer.benchmark` (random weights, 2 threads) in `results/benchmarks.json`; the continuous-batching numbers from `results/serving_cpu.json`, whose command and hardware are recorded in the file.

## Limits

- Measured speculative speedup and all GPU numbers are pending the T4 run.
- The CPU load test used a byte-level tokenizer (the Hugging Face hub was not reachable from the machine), so prompts were longer in tokens than GPT-2's BPE would make them; this affects prefill time equally for both schedulers.
- Prefill runs one request at a time; there is no chunked prefill, so a very long prompt delays the decoding step it joins.
- The KV cache reserves `max_seq_len` positions per slot (no paging).

## What I'd add next

- **Paged KV cache:** allocate cache in fixed-size blocks instead of reserving the maximum length per slot.
- **Chunked prefill:** split long prompts across iterations so decoding sequences are not stalled.
- **Quantized weights:** int8 weight-only linear layers to cut the memory reads that dominate decoding, checked by perplexity.
