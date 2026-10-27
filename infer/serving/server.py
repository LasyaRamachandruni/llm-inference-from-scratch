"""HTTP server: continuous batching behind a request queue, with streaming responses.

    uvicorn infer.serving.server:app --port 8000            # or: python -m infer.serving.server

    curl -N localhost:8000/v1/completions -H 'content-type: application/json' \\
         -d '{"prompt": "The history of the printing press", "max_tokens": 64, "stream": true}'

One background thread owns the model and runs the engine loop; request handlers
only enqueue work and read tokens back, so many concurrent HTTP requests share
each batched decoding step. Streaming uses server-sent events (one `data:` line
per token, then `data: [DONE]`).

Configuration (environment variables):
    INFER_MODEL             gpt2 | gpt2-medium | distilgpt2 ...      default gpt2
    INFER_PRETRAINED        1 = load trained weights from Hugging Face  default 0 (random weights)
    INFER_DEVICE            cpu | cuda | auto                         default auto
    INFER_POLICY            continuous | static                       default continuous
    INFER_MAX_SLOTS         sequences decoded together                default 16
    INFER_MAX_SEQ_LEN       prompt + output tokens per sequence       default 1024
    INFER_MAX_BATCH_TOKENS  tokens per scheduler iteration            default 2048
    INFER_MAX_QUEUE         waiting requests before 503               default 256
    INFER_DRAIN_SECONDS     on shutdown, time for open requests to finish  default 30

Health endpoints (used as Kubernetes probes):
    GET /health   liveness: 200 while the engine thread is alive, 500 if it died
    GET /ready    readiness: 200 once the model is loaded and the engine is running
    GET /metrics  JSON counters for people; GET /metrics/prometheus for Prometheus
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..model import GPT, config_for
from ..prompts import Tokenizer, load_tokenizer
from ..sampling import SamplingParams
from .engine import Engine, EngineConfig, Request
from .prom import ServerMetrics

log = logging.getLogger("infer.server")


@dataclass(frozen=True)
class ServerSettings:
    model: str = "gpt2"
    pretrained: bool = False
    device: str = "auto"
    policy: str = "continuous"
    max_slots: int = 16
    max_seq_len: int = 1024
    max_batch_tokens: int = 2048
    max_queue: int = 256

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ServerSettings:
        e = dict(os.environ) if env is None else env
        d = cls()
        return cls(
            model=e.get("INFER_MODEL", d.model),
            pretrained=e.get("INFER_PRETRAINED", "0").lower() in ("1", "true", "yes"),
            device=e.get("INFER_DEVICE", d.device),
            policy=e.get("INFER_POLICY", d.policy),
            max_slots=int(e.get("INFER_MAX_SLOTS", d.max_slots)),
            max_seq_len=int(e.get("INFER_MAX_SEQ_LEN", d.max_seq_len)),
            max_batch_tokens=int(e.get("INFER_MAX_BATCH_TOKENS", d.max_batch_tokens)),
            max_queue=int(e.get("INFER_MAX_QUEUE", d.max_queue)),
        )


def build_model(s: ServerSettings) -> GPT:
    device = torch.device(
        "cuda" if s.device == "auto" and torch.cuda.is_available() else ("cpu" if s.device == "auto" else s.device)
    )
    if s.pretrained:
        from ..weights import from_pretrained

        model = from_pretrained(s.model)
    else:
        torch.manual_seed(0)
        model = GPT(config_for(s.model))
    return model.to(device).eval()


class EngineRunner:
    """Owns the engine on one background thread; safe to call submit() from any thread.

    Only the engine thread touches the engine. Request handlers put new requests
    in a thread-safe inbox, which the engine thread moves into its queue between
    steps. (An earlier version shared a lock with the engine thread, which held it
    for a whole step and re-took it straight away; a handler waiting for it
    blocked the event loop for seconds under load, so health checks timed out.)
    """

    def __init__(self, engine: Engine, max_queue: int) -> None:
        self.engine = engine
        self.max_queue = max_queue
        self._inbox: queue.SimpleQueue[Request] = queue.SimpleQueue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self.started = time.time()
        self.errors = 0
        self.prom = ServerMetrics()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=10)

    def alive(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    def queue_depth(self) -> int:
        return len(self.engine.waiting) + self._inbox.qsize()

    def submit(self, req: Request) -> None:
        """Never blocks: validates, then hands the request to the engine thread."""
        if self.queue_depth() >= self.max_queue:
            self.prom.request_rejected("queue_full")
            raise OverflowError("queue full")
        self.engine.validate(req)  # ValueError -> 422, raised here in the handler
        self._inbox.put(req)
        self._wake.set()

    def _drain_inbox(self) -> None:
        while True:
            try:
                req = self._inbox.get_nowait()
            except queue.Empty:
                return
            self.engine.waiting.append(req)  # already validated

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._drain_inbox()
            if self.engine.idle:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            try:
                self.engine.step()
            except Exception:  # keep serving; the failing requests are ended below
                self.errors += 1
                log.exception("engine step failed; failing the running requests")
                for r in self.engine.running:
                    if not r.done:
                        self.engine.fail(r)
                self.engine.running.clear()
                self.engine.last_tokens.clear()
                self.engine.cache.lengths.zero_()

    def metrics(self) -> dict[str, Any]:
        st = self.engine.stats
        up = time.time() - self.started
        return {
            "policy": self.engine.config.policy,
            "waiting": self.queue_depth(),
            "running": len(self.engine.running),
            "completed": st.completed,
            "generated_tokens": st.generated_tokens,
            "steps": st.steps,
            "uptime_s": round(up, 1),
            "tokens_per_s_since_start": round(st.generated_tokens / up, 2) if up else 0.0,
            "engine_errors": self.errors,
        }

    def prometheus(self, max_slots: int) -> str:
        e = self.engine
        gauges = {
            "infer_queue_depth": ("Requests waiting for a decode slot", float(self.queue_depth())),
            "infer_requests_running": ("Requests being decoded", float(len(e.running))),
            "infer_slots": ("Decode slots (largest batch)", float(max_slots)),
            "infer_up": ("1 while the engine thread is alive", 1.0 if self.alive() else 0.0),
        }
        counters = {
            "infer_generated_tokens_total": ("Tokens generated", float(e.stats.generated_tokens)),
            "infer_engine_steps_total": ("Scheduler iterations", float(e.stats.steps)),
            "infer_engine_errors_total": ("Engine steps that raised", float(self.errors)),
        }
        return self.prom.render(gauges, counters)


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False
    ignore_eos: bool = False


class State:
    runner: EngineRunner
    tokenizer: Tokenizer
    settings: ServerSettings


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = ServerSettings.from_env()
    model = build_model(s)
    cfg = EngineConfig(s.max_slots, s.max_seq_len, s.max_batch_tokens, policy=s.policy)  # type: ignore[arg-type]
    state.settings = s
    state.tokenizer = load_tokenizer(s.model, allow_fallback=not s.pretrained)
    state.runner = EngineRunner(Engine(model, cfg), s.max_queue)
    state.runner.start()
    log.info(
        "serving %s (%s weights) with %s batching, %d slots",
        s.model,
        "trained" if s.pretrained else "random",
        s.policy,
        s.max_slots,
    )
    yield
    state.runner.stop()


app = FastAPI(title="llm-inference-from-scratch", version="0.2.0", lifespan=lifespan)


def _make_request(body: CompletionRequest, loop: asyncio.AbstractEventLoop) -> tuple[Request, asyncio.Queue]:
    ids = state.tokenizer.encode(body.prompt)
    queue: asyncio.Queue[tuple[int, bool]] = asyncio.Queue()

    def on_token(r: Request, tok: int, done: bool) -> None:
        if done:  # record before the handler can answer, so /metrics is never behind a response
            state.runner.prom.request_finished(r.finish_reason or "unknown", r.ttft, r.latency)
        loop.call_soon_threadsafe(queue.put_nowait, (tok, done))

    req = Request(
        prompt=ids,
        max_new_tokens=body.max_tokens,
        params=SamplingParams(body.temperature, body.top_k, body.top_p),
        eos_token_id=None if body.ignore_eos else state.tokenizer.eos_token_id,
        seed=body.seed,
        on_token=on_token,
    )
    try:
        state.runner.submit(req)
    except OverflowError as exc:  # Kubernetes autoscaling should keep this rare
        raise HTTPException(503, "server busy: request queue is full") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return req, queue


def _summary(req: Request) -> dict[str, Any]:
    return {
        "id": req.id,
        "prompt_tokens": len(req.prompt),
        "completion_tokens": len(req.output),
        "finish_reason": req.finish_reason,
        "ttft_ms": round(req.ttft * 1000, 2) if req.ttft is not None else None,
        "latency_ms": round(req.latency * 1000, 2) if req.latency is not None else None,
    }


@app.post("/v1/completions", response_model=None)
async def completions(body: CompletionRequest) -> StreamingResponse | dict[str, Any]:
    loop = asyncio.get_running_loop()
    req, queue = _make_request(body, loop)

    if body.stream:

        async def events() -> AsyncIterator[str]:
            index = 0
            while True:
                tok, done = await queue.get()
                if tok >= 0:
                    msg = {"index": index, "token_id": tok, "text": state.tokenizer.decode([tok])}
                    yield f"data: {json.dumps(msg)}\n\n"
                    index += 1
                if done:
                    yield f"data: {json.dumps({'done': True, **_summary(req)})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(events(), media_type="text/event-stream")

    while True:
        _, done = await queue.get()
        if done:
            break
    if req.finish_reason == "error":
        raise HTTPException(500, "generation failed")
    return {"text": state.tokenizer.decode(req.output), **_summary(req)}


@app.get("/health")
async def health() -> dict[str, Any]:
    s = state.settings
    if not state.runner.alive():  # liveness probe fails -> Kubernetes restarts the container
        raise HTTPException(500, "engine thread is not running")
    return {"status": "ok", "model": s.model, "pretrained": s.pretrained, "policy": s.policy}


@app.get("/ready")
async def ready() -> dict[str, str]:
    # The model loads in `lifespan` before the server accepts connections, so any
    # answer here means it is loaded; this still fails if the engine has stopped.
    if not state.runner.alive():
        raise HTTPException(503, "not ready")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    return state.runner.metrics()


@app.get("/metrics/prometheus", response_class=PlainTextResponse)
async def prometheus() -> PlainTextResponse:
    text = state.runner.prometheus(state.settings.max_slots)
    return PlainTextResponse(text, media_type="text/plain; version=0.0.4")


def main() -> None:
    import uvicorn

    # On SIGTERM uvicorn stops accepting connections and waits up to this long for
    # open requests (including streams) to finish before shutting the engine down.
    drain = int(os.environ.get("INFER_DRAIN_SECONDS", "30"))
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        timeout_graceful_shutdown=drain,
    )


if __name__ == "__main__":
    main()
