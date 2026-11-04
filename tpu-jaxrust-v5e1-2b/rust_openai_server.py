#!/usr/bin/env python3
"""OpenAI-compatible HTTP server over the Rust engine (`jaxrust_rs`).

    python3.13 rust_openai_server.py --context 2048 --port 8000

Serves `/v1/models`, `/health` and `/v1/chat/completions` (streamed or not) for
`google/gemma-4-E2B-it`. Decoding is greedy: the engine has no sampler, so
`temperature` and `top_k` are accepted and ignored. `ignore_eos` (vLLM's extension)
decodes the full `max_tokens`. A streamed reply ends with a usage chunk, the shape vLLM
sends for `stream_options.include_usage`.

`Engine` is bound to the thread that built it, so one worker thread owns it and runs
requests one at a time; HTTP handlers hand it jobs through a queue.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rust", "build"))

MODEL_ID = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it")
REVISION = os.environ.get("MODEL_REVISION", "4adf15748de9e1a0097b6274cee9c15836c41c37")
PLUGIN = os.environ.get(
    "PJRT_PLUGIN", "/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so"
)

app = FastAPI(title="Gemma 4 E2B on the Rust engine")
JOBS: queue.Queue = queue.Queue()
STATE = {"ready": False, "tokenizer": None, "stop": [], "error": None}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 128
    temperature: float | None = 0.0
    top_k: int | None = None
    stream: bool | None = False
    ignore_eos: bool | None = False


def _worker(model_dir: str, context: int, ready: threading.Event):
    import jaxrust_rs

    try:
        jaxrust_rs.init(PLUGIN)
        eng = jaxrust_rs.Engine(model_dir, context)
    except Exception as exc:  # noqa: BLE001 -- any load failure is reported by /health
        STATE["error"] = str(exc)
        ready.set()
        return
    STATE["engine_info"] = {
        "context": eng.context,
        "weight_bytes": eng.weight_bytes,
        **eng.timings(),
    }
    STATE["ready"] = True
    ready.set()
    while True:
        ids, max_new, stop, out = JOBS.get()

        def on_tokens(toks, out=out):
            out.put(("tokens", list(toks)))
            return True

        try:
            r = eng.decode(ids, max_new, stop=stop, on_tokens=on_tokens)
            out.put(("done", r))
        except Exception as exc:  # noqa: BLE001 -- handed to the request that caused it
            out.put(("error", str(exc)))


def _prompt_ids(messages) -> list[int]:
    tok = STATE["tokenizer"]
    out = tok.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    ids = [int(t) for t in out]
    if (
        not ids or ids[0] != tok.bos_token_id
    ):  # the JAX engine prepends <bos> the same way
        ids = [tok.bos_token_id] + ids
    return ids


def _submit(req: ChatCompletionRequest):
    if not STATE["ready"]:
        raise HTTPException(
            status_code=503, detail=STATE["error"] or "Rust engine is loading"
        )
    ids = _prompt_ids(req.messages)
    out: queue.Queue = queue.Queue()
    JOBS.put((ids, req.max_tokens or 128, [] if req.ignore_eos else STATE["stop"], out))
    return ids, out


def _chunk(req_id, created, model, choices, usage=None):
    d = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        d["usage"] = usage
    return f"data: {json.dumps(d)}\n\n"


@app.get("/health")
def health():
    return {
        "ready": STATE["ready"],
        "error": STATE["error"],
        "model": MODEL_ID,
        **STATE.get("engine_info", {}),
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "jaxrust"}],
    }


@app.post("/v1/chat/completions")
def chat(req: ChatCompletionRequest):
    ids, out = _submit(req)
    tok = STATE["tokenizer"]
    model = req.model or MODEL_ID
    created = int(time.time())
    req_id = f"chatcmpl-rust-{int(time.time() * 1000)}"

    if req.stream:

        def gen():
            while True:
                kind, val = out.get()
                if kind == "tokens":
                    for t in val:
                        text = tok.decode([t], skip_special_tokens=True)
                        yield _chunk(
                            req_id,
                            created,
                            model,
                            [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        )
                elif kind == "done":
                    finish = "length" if val.finish == "length" else "stop"
                    yield _chunk(
                        req_id,
                        created,
                        model,
                        [{"index": 0, "delta": {}, "finish_reason": finish}],
                    )
                    n = len(val.tokens)
                    yield _chunk(
                        req_id,
                        created,
                        model,
                        [],
                        {
                            "prompt_tokens": len(ids),
                            "completion_tokens": n,
                            "total_tokens": len(ids) + n,
                        },
                    )
                    yield "data: [DONE]\n\n"
                    return
                else:
                    yield f"data: {json.dumps({'error': {'message': val}})}\n\n"
                    return

        return StreamingResponse(gen(), media_type="text/event-stream")

    while True:
        kind, val = out.get()
        if kind == "done":
            break
        if kind == "error":
            raise HTTPException(status_code=500, detail=val)
    n = len(val.tokens)
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": tok.decode(val.tokens, skip_special_tokens=True),
                },
                "finish_reason": "length" if val.finish == "length" else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(ids),
            "completion_tokens": n,
            "total_tokens": len(ids) + n,
            "prefill_s": val.prefill_s,
            "decode_s": val.decode_s,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--context", type=int, default=2048, help="KV cache length, a multiple of 128"
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_dir = snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"],
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    STATE["tokenizer"] = tok
    stop = {tok.eos_token_id}
    unk = tok.unk_token_id
    for name in ("<end_of_turn>", "<turn|>"):
        t = tok.convert_tokens_to_ids(name)
        if isinstance(t, int) and t >= 0 and t != unk:
            stop.add(t)
    STATE["stop"] = sorted(s for s in stop if isinstance(s, int))

    ready = threading.Event()
    threading.Thread(
        target=_worker, args=(model_dir, args.context, ready), daemon=True
    ).start()
    ready.wait()
    print("engine:", STATE.get("engine_info") or STATE["error"], flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
