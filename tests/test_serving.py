"""Continuous batching engine, HTTP server, load-test helpers and statistics."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator

import pytest
import torch

from infer import GPT, GPTConfig, SamplingParams, generate
from infer.prompts import ByteTokenizer, bundled, load, output_lengths
from infer.serving.engine import Engine, EngineConfig, Request
from infer.stats import bootstrap_ci, percentile, summarize


def tiny(seed: int = 0, vocab: int = 97) -> GPT:
    torch.manual_seed(seed)
    m = GPT(GPTConfig(vocab_size=vocab, n_positions=128, n_embd=32, n_layer=2, n_head=4))
    return m.double().eval()


def trace(n: int = 25, seed: int = 0) -> list[tuple[list[int], int]]:
    rng = random.Random(seed)
    return [(rng.choices(range(97), k=rng.randint(1, 20)), rng.randint(1, 30)) for _ in range(n)]


# --- engine -------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["continuous", "static"])
def test_engine_matches_single_request_generation(policy: str) -> None:
    """Requests joining and leaving mid-batch get exactly the tokens they would get alone."""
    m = tiny()
    reqs = trace()
    engine = Engine(m, EngineConfig(max_slots=4, max_seq_len=128, max_batch_tokens=40, policy=policy))  # type: ignore[arg-type]
    submitted = []
    for i, (p, n) in enumerate(reqs):
        submitted.append(engine.submit(Request(p, n)))
        if i % 3 == 0:
            engine.step()  # arrivals interleaved with decoding
    engine.run_until_done()
    for r, (p, n) in zip(submitted, reqs, strict=True):
        assert r.output == generate(m, p, n), r.id
        assert r.finish_reason == "length" and r.ttft is not None and r.latency is not None


def test_continuous_batching_wastes_no_rows_and_takes_fewer_steps() -> None:
    m = tiny()
    stats = {}
    for policy in ("continuous", "static"):
        e = Engine(m, EngineConfig(max_slots=4, max_seq_len=128, policy=policy))  # type: ignore[arg-type]
        for p, n in trace():
            e.submit(Request(p, n))
        e.run_until_done()
        stats[policy] = e.stats
    useful = stats["continuous"].generated_tokens - stats["continuous"].prefills
    assert stats["continuous"].computed_rows == useful  # every computed row produced a token
    assert stats["static"].computed_rows > useful
    assert stats["continuous"].steps < stats["static"].steps


def test_slots_and_token_budget_are_respected() -> None:
    m = tiny()
    e = Engine(m, EngineConfig(max_slots=3, max_seq_len=128, max_batch_tokens=25))
    for _ in range(6):
        e.submit(Request([1] * 10, 5))
    e.step()
    assert len(e.running) == 2  # a third 10-token prefill would exceed 25 tokens this iteration
    while not e.idle:
        e.step()
        assert len(e.running) <= 3


def test_eos_stops_a_request_early() -> None:
    m = tiny()
    first = generate(m, [3, 4, 5], 10)
    stop = first.index(first[2])  # first occurrence of the token used as EOS
    e = Engine(m, EngineConfig(max_slots=2, max_seq_len=64))
    r = e.submit(Request([3, 4, 5], 10, eos_token_id=first[2]))
    e.run_until_done()
    assert r.output == first[: stop + 1] and r.finish_reason == "eos"


def test_a_failing_prefill_ends_only_that_request() -> None:
    m = tiny()
    events: list[tuple[int, bool]] = []
    e = Engine(m, EngineConfig(max_slots=2, max_seq_len=64))
    bad = e.submit(Request([500], 3, on_token=lambda _r, t, d: events.append((t, d))))  # id outside the vocabulary
    good = e.submit(Request([1, 2], 3))
    e.run_until_done()
    assert bad.finish_reason == "error" and events == [(-1, True)]
    assert good.output == generate(m, [1, 2], 3)


def test_sampled_requests_are_reproducible_with_a_seed() -> None:
    m = tiny()
    params = SamplingParams(temperature=1.0)
    outs = []
    for _ in range(2):
        e = Engine(m, EngineConfig(max_slots=2, max_seq_len=64))
        a = e.submit(Request([1, 2, 3], 12, params=params, seed=7))
        e.submit(Request([9, 8], 5, params=params, seed=1))
        e.run_until_done()
        outs.append(a.output)
    assert outs[0] == outs[1]


def test_invalid_requests_are_rejected() -> None:
    e = Engine(tiny(), EngineConfig(max_slots=2, max_seq_len=32, max_batch_tokens=16))
    with pytest.raises(ValueError, match="max_seq_len"):
        e.submit(Request([1] * 10, 30))
    with pytest.raises(ValueError, match="empty"):
        e.submit(Request([], 3))
    with pytest.raises(ValueError, match="max_batch_tokens"):
        e.submit(Request([1] * 20, 3))
    with pytest.raises(ValueError, match="context window"):
        Engine(tiny(), EngineConfig(max_seq_len=4096))


def test_token_callback_streams_every_token() -> None:
    seen: list[tuple[int, bool]] = []
    e = Engine(tiny(), EngineConfig(max_slots=2, max_seq_len=64))
    r = e.submit(Request([1, 2], 4, on_token=lambda _r, t, d: seen.append((t, d))))
    e.run_until_done()
    assert [t for t, _ in seen] == r.output and [d for _, d in seen] == [False, False, False, True]


# --- server -------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    from fastapi.testclient import TestClient

    import infer.serving.server as server

    monkeypatch.setattr(server, "build_model", lambda s: tiny(vocab=300).float())  # byte ids < 256
    monkeypatch.setattr(server, "load_tokenizer", lambda name, allow_fallback=True: ByteTokenizer())
    monkeypatch.setenv("INFER_MAX_SEQ_LEN", "128")
    with TestClient(server.app) as c:
        yield c


def test_server_completion_and_metrics(client: object) -> None:
    resp = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 6})  # type: ignore[attr-defined]
    assert resp.status_code == 200
    body = resp.json()
    assert body["completion_tokens"] == 6 and body["prompt_tokens"] == 5 and body["finish_reason"] == "length"
    m = client.get("/metrics").json()  # type: ignore[attr-defined]
    assert m["completed"] == 1 and m["generated_tokens"] == 6
    assert client.get("/health").json()["status"] == "ok"  # type: ignore[attr-defined]


def test_server_streams_tokens_as_server_sent_events(client: object) -> None:
    with client.stream("POST", "/v1/completions", json={"prompt": "hi", "max_tokens": 5, "stream": True}) as resp:  # type: ignore[attr-defined]
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    tokens = [json.loads(line[6:]) for line in lines[:-2]]
    assert [t["index"] for t in tokens] == [0, 1, 2, 3, 4]
    assert json.loads(lines[-2][6:])["done"] is True and lines[-1] == "data: [DONE]"


def test_server_validates_input(client: object) -> None:
    assert client.post("/v1/completions", json={"prompt": "", "max_tokens": 5}).status_code == 422  # type: ignore[attr-defined]
    assert client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0}).status_code == 422  # type: ignore[attr-defined]
    too_long = client.post("/v1/completions", json={"prompt": "x" * 100, "max_tokens": 100})  # type: ignore[attr-defined]
    assert too_long.status_code == 422 and "max_seq_len" in too_long.json()["detail"]


# --- load test helpers and statistics -----------------------------------------------------


def test_offline_load_test_runs_both_policies_on_the_same_trace() -> None:
    from infer.serving.loadtest import TraceItem, run_offline

    m = tiny()
    items = [TraceItem(p, n, 0.0) for p, n in trace(12)]
    out = {
        pol: run_offline(m, items, EngineConfig(max_slots=4, max_seq_len=128, policy=pol))
        for pol in ("continuous", "static")
    }  # type: ignore[arg-type]
    assert out["continuous"]["output_tokens"] == out["static"]["output_tokens"] == sum(n for _, n in trace(12))
    assert out["continuous"]["wasted_row_share"] == pytest.approx(0.0)
    assert out["static"]["wasted_row_share"] > 0


def test_bundled_prompts_and_output_lengths() -> None:
    prompts = bundled()
    assert len(prompts) == 60 and len({p.text for p in prompts}) == 60
    assert load("bundled", n=5) == prompts[:5]
    lengths = output_lengths(500, 16, 256, seed=0)
    assert min(lengths) >= 16 and max(lengths) <= 256 and lengths == output_lengths(500, 16, 256, seed=0)
    with pytest.raises(ValueError, match="unknown"):
        load("sharegpt")


def test_byte_tokenizer_round_trip() -> None:
    tok = ByteTokenizer()
    assert tok.decode(tok.encode("héllo")) == "héllo"


def test_summary_statistics() -> None:
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5 and s["n"] == 4 and s["ci_low"] <= 2.5 <= s["ci_high"]
    assert bootstrap_ci([5.0]) == (5.0, 5.0)
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([0, 10], 90) == 9


def test_load_test_trace_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    import infer.serving.loadtest as lt

    monkeypatch.setattr(lt, "load_tokenizer", lambda name: ByteTokenizer())
    items = lt.build_trace("gpt2", 10, 4, 8, rate=5.0, source="bundled", seed=0, max_prompt=32)
    assert len(items) == 10 and all(len(t.prompt) <= 32 and 4 <= t.max_new_tokens <= 8 for t in items)
    assert [t.arrival_s for t in items] == sorted(t.arrival_s for t in items) and items[-1].arrival_s > 0

    m = tiny(vocab=300)
    runs = {
        p: [lt.run_offline(m, items, EngineConfig(max_slots=3, max_seq_len=64, policy=p))]
        for p in ("continuous", "static")
    }  # type: ignore[arg-type]
    summary = {
        p: {
            k: summarize([r[src] if isinstance(r[src], float) else r[src]["mean"] for r in rs])
            for k, src in [
                ("tokens_per_s", "tokens_per_s"),
                ("latency_mean_s", "latency_s"),
                ("latency_p99_s", "latency_s"),
                ("ttft_mean_s", "ttft_s"),
                ("wasted_row_share", "wasted_row_share"),
            ]
        }
        for p, rs in runs.items()
    }
    result = {
        "model": "tiny",
        "weights": "random",
        "hardware": "cpu",
        "torch": torch.__version__,
        "trace": {"requests": 10, "output_tokens": "4-8", "arrival_rate_per_s": 5.0},
        "engine": {"max_slots": 3},
        "summary": summary,
        "runs": runs,
        "throughput_ratio_continuous_over_static": 1.0,
    }
    table = lt.markdown(result)
    assert "| static |" in table and "| continuous |" in table
