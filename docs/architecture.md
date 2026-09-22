# Architecture and operational choices

The API keeps Decider's typed decision contract: a state plus named `choice`, `noul` or `score` questions. It does not generate a tool argument payload. The gateway can constrain the next tool selection while Codex remains responsible for arguments and execution.

## Memory lifecycle

| Phase | Model location | Accepts inference? |
| --- | --- | --- |
| `loading` | Loading/quantizing in RAM | No; gateway passthrough |
| `idle`, no previous GPU request | RAM; observed WDDM 0 MiB | Yes; first request initializes the GPU |
| `resuming` | Moving to GPU | Current request only |
| `ready` | GPU | One request at a time |
| `sleeping` | Moving to RAM | No; concurrent requests get 503 |
| `idle`, previously used GPU | RAM; driver allocations may remain | Yes; approximately 0.5 s resume in the measured short case |
| `recycle_pending` | RAM, awaiting process termination | No; gateway passthrough |
| `failed` | Unavailable | No; supervisor recovery |

The GPU worker is a single `ThreadPoolExecutor`. A cancelled HTTP coroutine does not clear the busy flag before its GPU future finishes. Idle timers start after GPU work finishes, not after the HTTP client times out.

At 10 seconds idle, weights move to RAM and PyTorch's allocator cache is released. WDDM still retained about 2.5 GiB in one measured state even though PyTorch reported only 76 MiB. At 30 seconds idle the service asks the supervisor to replace its process. The supervisor checks every 10 seconds, so full release normally occurs around 30–40 seconds. OS scheduling and shutdown can add delay.

The replacement loads in RAM and does not recycle again until it has actually used the GPU. Planned recycling is distinguished from crashes, including when the child process takes an extra supervisor iteration to exit. Unplanned failures are bounded to three starts in ten minutes.

## INT8 storage, BF16 computation

138 large linear layers were converted in the measured checkpoint. Each output channel gets a scale; signed 8-bit values store its weights. Each forward reconstructs that layer's BF16 weight just before `linear`. Embedding, output head and smaller layers keep their original precision.

This saves persistent weight storage. It is not a native INT8 GEMM accelerator. The source checkpoint is not rewritten, and calibration probabilities can change. `bf16` remains an optional storage mode.

The upstream FP8 implementation uses `torch._scaled_mm` with rowwise scaling and describes Hopper support. Three actual matrix shapes failed on the measured ROCm build with a CUDA-only rowwise-scaling error. This does not mean that RDNA4 hardware lacks FP8 capability.

A generic GGUF generation server does not directly supply this implementation's slot-logit/typed-question interface. A GGUF alternative would need its own compatible decision backend and validation; none is claimed here.

## Budgets and fallback

| Setting | Default |
| --- | ---: |
| State token cap | 4,096 |
| Sequence token cap | 6,144 |
| Estimated total work token cap | 12,288 |
| Forward grouping token budget | 4,096 |
| Gateway state characters | 6,000 |
| Gateway characters per message | 1,500 |
| Minimum choice confidence | 0.80 |
| Gateway timeout per attempt | 2,000 ms |
| Gateway retry | Once |

Input validation occurs before GPU upload. The token estimates are conservative and include a question allowance. Padding preserves masks and answer positions. Inputs beyond the budget are rejected rather than silently dropping their latest request. These local decisions do not truncate the main Codex upstream request.

The independent `needs_tool` question receives the same tool definitions as the choice question. Neither probabilities nor confidence values are patched after inference. Disagreement, low confidence, tool-free responses, memory pressure and backend errors fall back through the gateway. A long cold inference can continue in the background after the gateway stops waiting.

## Windows operation

Both HTTP services bind to loopback. The local backend uses a placeholder credential because its requests remain on the machine; it is not designed as an authenticated network API. Keep the binding on `127.0.0.1`.

The scheduled task runs as the current interactive user without elevation. A named Windows mutex prevents duplicate supervisors. Shutdown targets the PID reported by the local service identity, and an unexpected service on the reserved ports causes an error.

Runtime configuration, logs, pause markers and backups live outside this repository. The installer does not alter model weights, GPU packages, Codex authentication, or unrelated processes.

## Recovery

- `jev-local.ps1 stop` pauses the local service persistently. `start` clears the pause marker.
- `restart` reloads precision or idle settings from `installation.json`.
- A newly initialized Triton process can take tens of seconds on its first request. The gateway's timeout still applies.
- If the backend fails repeatedly, inspect `local-service.log` and `decider.log` in the gateway state directory.
- Each installer run backs up overwritten files and, when replacing an existing scheduled task, its task XML. Restore only the installation's own files after stopping it.
- For immediate VRAM release, use `stop`. RAM offload alone is not proof that Windows has returned all dedicated GPU memory.
