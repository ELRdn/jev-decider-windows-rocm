# Measurements and validation

Measured on **2026-09-22**, Windows, **AMD Radeon RX 9070 XT / gfx1201**, 15.92 GiB VRAM. [Machine-readable, sanitized results](../benchmarks/2026-09-22-rx9070xt.json).

## Environment

| Component | Observed version |
| --- | --- |
| Python | 3.12 |
| Jev Gateway | 0.4.1 |
| Decider package / model | 0.9.0 / `decider-v10` |
| Decider source | `c4daaac28af9fea95d627015cffa2dd5a5926ee6` |
| PyTorch | 2.13.0+rocm10.0.0 |
| HIP | 7.15.26333 |
| Transformers | 5.17.0 |
| Triton Windows | 3.7.1.post27 |
| Flash Linear Attention | 0.5.2 |

These are observed versions of an existing custom environment, not a portable lockfile or a claim that all wheels are available on public PyPI. Real GPU kernels must work on the selected architecture before installing this integration. Graphs and `torch.compile` were disabled; causal convolution used the reference PyTorch fallback.

## Memory

| Measurement | MiB |
| --- | ---: |
| Old resident Jev process, Windows dedicated GPU memory | 6,331 |
| New service, active/resident samples | 3,187.2–3,790.2 |
| New service after RAM offload, Windows dedicated GPU memory | 2,603.7 |
| New service after process replacement | **0** |
| Old/new PyTorch allocation | 3,665.32 / 2,370.4 |
| Whole adapter before / after full idle release | 14,222 / 7,891.9 |
| RAM standby process working set / private memory | 3,089.4 / 4,125.5 |

WDDM adapter/process counters and PyTorch allocator statistics measure different things. Process totals can also include shared resources; adding them is not a reliable adapter total. Active values are samples after inference, not continuous maximum measurements. Other applications were left running and affect the adapter totals.

## Latency and lifecycle

| Probe | Result |
| --- | --- |
| 52-token classification, 10 warm calls | 10 correct; median 74.82 ms |
| Two questions, 1,866 input tokens | Warm call 451.91 ms |
| Two questions, 3,434 input tokens | Warm call 954.06 ms |
| Resume from RAM, then short inference | 495.94 ms |
| First GPU inference in a fresh process | **25,781.08 ms** |
| Last inference to observed process replacement | **39.5 s** |
| Three simultaneous requests | One 200, two 503; subsequent request succeeded |
| Invalid/oversized input while in RAM | 422/413, no GPU upload |
| Gateway English read/write dry decisions | Correct tool selected |
| Gateway Japanese read request | Correct choice; independent disagreement delegated to Codex |
| Gateway no-tool request | Passed through to Codex |
| Sign-in task and deep idle | Task launched; GPU PID replaced; CPU-ready successor had WDDM 0 MiB |

Warm/cold cases are distinct. The RAM-resume number applies before full process replacement. The gateway timeout is 2 seconds per attempt with one retry, so it does not wait the full cold inference time before falling back. The local direct API does wait for the inference when its client timeout permits.

## Quantization comparison

The same v10 checkpoint was evaluated in BF16 and INT8 storage on 22 synthetic English/Japanese cases: billing/technical/sales classification, file read/write, shell/test, web lookup, direct response, already-completed actions, ambiguity and longer contexts. There were 37 total questions.

- All 22 choice labels matched.
- Choice confidence bins at 0.80 matched.
- Noul bins below 0.30, from 0.30 to below 0.70, and at least 0.70 matched. These bins are an explicit comparison metric, not a complete simulation of the gateway policy.
- Maximum absolute choice-probability difference: **0.0459**.
- Maximum absolute Noul difference: **0.0346**.

This is a small regression sample, not proof of general accuracy retention. The checkpoint is the same; numeric outputs are not identical. No 0.8B model was downloaded or evaluated.

## What the repository tests cover

`tests/test_low_memory.py` checks CPU-only startup, first request activation, idle offload races, cancellation while GPU work continues, initialization failure, deep-idle transitions, input validation, question preservation and optional Decider padding.

`tests/test_manager.py` checks that multiple planned recycles with delayed child exit do not consume the crash restart limit. `tests/test_weight_only.py` checks close numerical outputs, dimensions, dtype, bias, zero weights and untouched small layers/embeddings on CPU. The installer test uses an isolated fixture and mocked system-state inspection; it never installs a task or starts an actual GPU backend.

The published source adds portable paths and `uvicorn --app-dir` to the measured local implementation. Packaging tests do not substitute for the earlier hardware run. CI validates Windows CPU behavior and PowerShell syntax, with no ROCm hardware present.

## Unverified

End-to-end Codex task acceleration, token savings, broad multilingual quality, other GPU architectures, continuous peak VRAM, a complete fresh-PC ROCm installation, and a GGUF equivalent are not established by these measurements.

The removed resident startup had warmed contexts up to roughly 14k tokens. The new service uses smaller budgets and delegates longer requests; the old long-context measurements are not performance claims for this configuration.

## Sources

- [Decider](https://github.com/Mapika/decider), especially [`fp8.py`](https://github.com/Mapika/decider/blob/c4daaac28af9fea95d627015cffa2dd5a5926ee6/decider/fp8.py) and [`model.py`](https://github.com/Mapika/decider/blob/c4daaac28af9fea95d627015cffa2dd5a5926ee6/decider/model.py)
- [Decider-2B weights](https://huggingface.co/Mapika/decider-2b)
- [Jev Gateway](https://github.com/vinilana/jev-gateway)
