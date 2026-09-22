"""Local Decider-2B service for Jev Gateway on Windows/ROCm.

Run one worker only: python -m uvicorn local_rocm_server:app --host 127.0.0.1 --port 8000
"""
import asyncio
import copy
import gc
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# These must be set BEFORE importing torch, transformers, FLA or Decider.
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(os.environ["LOCALAPPDATA"]) / "TritonCache" / "decider-jev"))
os.environ["FLA_CACHE_RESULTS"] = "0"
os.environ["FLA_CACHE_MODE"] = "disabled"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODEL_ID = "Mapika/decider-2b"
MAX_STATE_TOKENS = 4096
MAX_SEQUENCE_TOKENS = 6144
MAX_WORK_TOKENS = 12288
MAX_FORWARD_TOKENS = 4096
IDLE_SECONDS = float(os.environ.get("JEV_LOCAL_IDLE_SECONDS", "10"))
DEEP_IDLE_SECONDS = float(os.environ.get("JEV_LOCAL_DEEP_IDLE_SECONDS", "30"))
PRECISION = os.environ.get("JEV_LOCAL_PRECISION", "int8")
logger = logging.getLogger("uvicorn.error")


class SystemOneRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=16)
    model: str | None = None
    independent: bool = True


def prepare_questions(questions):
    """Give the independent yes/no question the same tool definitions as Choice.

    Model answers and confidence values are never rewritten.
    """
    result = copy.deepcopy(questions)
    tool, needs = result.get("tool", {}), result.get("needs_tool", {})
    criteria = tool.get("criteria")
    if tool.get("type") == "choice" and needs.get("type") == "noul" and isinstance(criteria, dict):
        available = {name: description for name, description in criteria.items() if name != "no_tool_needed"}
        needs["instructions"] = (
            "Available tools: " + json.dumps(available, ensure_ascii=False) + "\n"
            "Does fulfilling the user's latest request require calling one of these tools now? "
            "Answer yes when a necessary action or information lookup has not been performed yet. "
            "Answer no when the assistant can already answer from the conversation."
        )
    return result


def check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Model produced a non-finite value")
    if isinstance(value, dict):
        for child in value.values():
            check_finite(child)
    elif isinstance(value, list):
        for child in value:
            check_finite(child)


def install_shape_buckets():
    """Bound Triton specialization to power-of-two lengths, preserving all masks/slots.

    This affects this service process only; the installed Decider package is untouched.
    """
    import decider.infer as infer
    import torch.nn.functional as functional

    original = infer.collate
    if getattr(original, "jev_bucketed", False):
        return

    def bucketed(items, pad_id):
        batch = original(items, pad_id)
        length = batch["input_ids"].shape[1]
        target = min(MAX_SEQUENCE_TOKENS, 1 << (length - 1).bit_length())
        if target > length:
            batch["input_ids"] = functional.pad(batch["input_ids"], (0, target - length), value=pad_id)
            batch["attention_mask"] = functional.pad(batch["attention_mask"], (0, target - length), value=0)
        return batch

    bucketed.jev_bucketed = True
    infer.collate = bucketed


def validate_request(model, req):
    from decider.systemone import plan_rows, render_question, render_state

    if req.model not in (None, "jev-latest", "decider-2b", "decider-v10", MODEL_ID):
        raise HTTPException(422, "Unsupported local model")
    questions = prepare_questions(req.questions)
    try:
        rendered = {key: render_question(spec) for key, spec in questions.items()}
        rows, _ = plan_rows(rendered, model.isolated_levels and req.independent)
        state_tokens = len(model.m.tok.encode("Context:\n" + render_state(req.state), add_special_tokens=False))
        question_tokens = [len(model.m.tok.encode(row["question"] + "\n" + "\n".join(row["options"]), add_special_tokens=False)) + 1024 for row in rows]
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(422, "Invalid question schema") from exc
    sequence_tokens = state_tokens + (max(question_tokens) if req.independent else sum(question_tokens))
    work_tokens = sum(state_tokens + size for size in question_tokens) if req.independent else sequence_tokens
    if state_tokens > MAX_STATE_TOKENS or sequence_tokens > MAX_SEQUENCE_TOKENS or work_tokens > MAX_WORK_TOKENS:
        # Decider's native truncation keeps the beginning and can lose the latest request.
        # Reject instead, so Jev Gateway leaves this request to Codex unchanged.
        raise HTTPException(413, "Request exceeds the local inference budget; use the main model")
    return questions


def run_prepared(model, req, questions):
    result = model.system_one(req.state, questions, independent=req.independent,
                              max_state_tokens=MAX_STATE_TOKENS, max_fwd_tokens=MAX_FORWARD_TOKENS)
    check_finite(result)
    return result


def run_inference(model, req):
    return run_prepared(model, req, validate_request(model, req))


def load_model(runtime):
    import importlib.metadata
    import torch
    from decider.infer import Decider
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # required fast path
    from jev_weight_only import convert_weight_only

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("The configured environment must provide a ROCm GPU")
    runtime.hardware = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "hip": torch.version.hip, "fla": importlib.metadata.version("flash-linear-attention"),
                        "triton": importlib.metadata.version("triton-windows")}
    install_shape_buckets()
    torch.set_num_threads(8)
    if PRECISION not in ("int8", "bf16"):
        raise ValueError("JEV_LOCAL_PRECISION must be int8 or bf16")
    # Sign-in loads the checkpoint into RAM only. No GPU warmup or BF16 GPU copy.
    model = Decider(MODEL_ID, device="cpu", use_graphs=False)
    if model.name != "decider-v10":
        raise RuntimeError("Expected the previously verified Decider v10 checkpoint")
    model.m.lm.config.use_cache = False
    model.m.lm.model.config.use_cache = False
    runtime.quantization = {"format": "bf16", "layers": 0, "saved_mib": 0}
    if PRECISION == "int8":
        runtime.quantization = convert_weight_only(model.m.lm.model, "int8")
    runtime.weight_bytes = sum(t.numel() * t.element_size() for t in list(model.m.parameters()) + list(model.m.buffers()))
    gc.collect()
    return model


def gpu_memory():
    import torch
    if not torch.cuda.is_initialized():
        return {"allocated_vram_mib": 0, "reserved_vram_mib": 0, "peak_allocated_vram_mib": 0}
    return {"allocated_vram_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            "reserved_vram_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
            "peak_allocated_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def release_gpu(runtime):
    import torch
    started = time.perf_counter()
    runtime.model.m.to("cpu")
    runtime.model.dev = "cpu"
    gc.collect()
    torch.cuda.empty_cache()
    runtime.stats["offloads"] += 1
    runtime.stats["last_offload_ms"] = round((time.perf_counter() - started) * 1000, 2)


def execute_request(runtime, req):
    import torch
    # Invalid/oversized requests must not wake the GPU.
    questions = validate_request(runtime.model, req)
    needs_upload = runtime.model.dev == "cpu"
    free, _ = torch.cuda.mem_get_info()
    required = (runtime.weight_bytes if needs_upload else 0) + 1024 * 2**20
    if free < required:
        runtime.stats["rejected_memory"] += 1
        raise HTTPException(503, "Insufficient free GPU memory; use the main model")
    try:
        if needs_upload:
            runtime.phase = "resuming"
            started = time.perf_counter()
            runtime.model.m.to("cuda")
            runtime.model.dev = "cuda"
            runtime.stats["resumes"] += 1
            runtime.stats["last_resume_ms"] = round((time.perf_counter() - started) * 1000, 2)
        runtime.phase = "ready"
        result = run_prepared(runtime.model, req, questions)
        # Release temporary dequantized weights and inference workspace immediately.
        torch.cuda.empty_cache()
        return result
    except torch.cuda.OutOfMemoryError as exc:
        runtime.stats["rejected_memory"] += 1
        release_gpu(runtime)
        runtime.phase = "idle"
        raise HTTPException(503, "GPU memory pressure; use the main model") from exc
    finally:
        runtime.last_used = time.monotonic()


class Runtime:
    def __init__(self):
        self.phase = "loading"
        self.hardware = {}
        self.model = None
        self.quantization = {}
        self.weight_bytes = 0
        self.busy = False
        self.inflight_started = None
        self.started = time.monotonic()
        self.last_used = self.started
        self.stats = {"completed": 0, "rejected_busy": 0, "rejected_input": 0, "rejected_memory": 0,
                      "errors": 0, "last_ms": None, "resumes": 0, "offloads": 0,
                      "last_resume_ms": None, "last_offload_ms": None}
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decider-gpu")

    async def initialize(self):
        try:
            self.model = await asyncio.get_running_loop().run_in_executor(self.worker, load_model, self)
            self.phase = "idle"
            logger.info("Decider v10 ready in RAM after %.1f seconds (%s); GPU loads on demand",
                        time.monotonic() - self.started, self.quantization.get("format"))
        except Exception:
            self.phase = "failed"
            logger.exception("Decider initialization failed")

    async def idle_loop(self):
        while True:
            await asyncio.sleep(.5)
            if (self.phase == "idle" and not self.busy and self.stats["resumes"] > 0
                    and time.monotonic() - self.last_used >= DEEP_IDLE_SECONDS):
                # ROCm/WDDM retains driver allocations even after empty_cache().
                # The supervisor replaces this process with a fresh CPU-only one.
                self.phase = "recycle_pending"
                logger.info("Deep idle: requesting process recycle to release WDDM GPU allocations")
                continue
            if self.phase != "ready" or self.busy or time.monotonic() - self.last_used < IDLE_SECONDS:
                continue
            self.busy = True
            self.phase = "sleeping"
            try:
                await asyncio.get_running_loop().run_in_executor(self.worker, release_gpu, self)
                self.phase = "idle"
                logger.info("Idle GPU weights released to RAM")
            except Exception:
                self.phase = "failed"
                logger.exception("GPU offload failed")
            finally:
                self.busy = False


@asynccontextmanager
async def lifespan(app):
    runtime = Runtime()
    app.state.runtime = runtime
    init_task = asyncio.create_task(runtime.initialize())
    idle_task = asyncio.create_task(runtime.idle_loop())
    yield
    for task in (init_task, idle_task):
        task.cancel()
    await asyncio.gather(init_task, idle_task, return_exceptions=True)
    runtime.worker.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Local Jev / Decider", lifespan=lifespan)


@app.get("/health")
async def health():
    runtime = app.state.runtime
    ready = runtime.phase in ("idle", "ready", "resuming", "sleeping")
    return {"ok": ready, "ready": ready, "phase": runtime.phase,
            "service": "jev-local-decider", "pid": os.getpid(), "backend": "local-rocm", "model": MODEL_ID,
            "busy": runtime.busy, "uptime_seconds": round(time.monotonic() - runtime.started),
            "gpu_resident": runtime.model is not None and runtime.model.dev == "cuda",
            "idle_release_seconds": IDLE_SECONDS, "quantization": runtime.quantization,
            "deep_idle_seconds": DEEP_IDLE_SECONDS,
            "inflight_seconds": round(time.monotonic() - runtime.inflight_started, 1) if runtime.inflight_started else None,
            "hardware": {**runtime.hardware, **gpu_memory()}, "stats": runtime.stats,
            "max_state_tokens": MAX_STATE_TOKENS, "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "disk_autotune_cache": False}


@app.get("/v1/models")
async def models():
    return {"models": [{"name": "decider-2b", "description": "Local Decider-2B on AMD ROCm"}]}


@app.post("/v1/systemone")
async def system_one(req: SystemOneRequest):
    runtime = app.state.runtime
    if runtime.phase not in ("idle", "ready", "resuming", "sleeping"):
        raise HTTPException(503, "Local model is loading or unavailable", headers={"Retry-After": "1"})
    if runtime.busy:
        runtime.stats["rejected_busy"] += 1
        # Do not enqueue requests that will outlive the gateway timeout and its retry.
        raise HTTPException(503, "Local GPU is busy; use the main model", headers={"Retry-After": "1"})
    runtime.busy = True
    runtime.inflight_started = time.monotonic()
    start = time.perf_counter()
    future = asyncio.get_running_loop().run_in_executor(runtime.worker, execute_request, runtime, req)

    def finished(_):
        # Keep busy until GPU work really ends, even if the HTTP caller disconnects.
        runtime.busy = False
        runtime.inflight_started = None
        runtime.stats["last_ms"] = round((time.perf_counter() - start) * 1000, 2)

    future.add_done_callback(finished)
    try:
        result = await asyncio.shield(future)
        runtime.stats["completed"] += 1
        return result
    except HTTPException as exc:
        if exc.status_code in (413, 422):
            runtime.stats["rejected_input"] += 1
        raise
    except Exception:
        runtime.stats["errors"] += 1
        runtime.phase = "failed"
        logger.exception("Local inference failed")
        return JSONResponse(status_code=503, content={"detail": "Local inference failed; use the main model"})
