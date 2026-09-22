import json
import os
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from decider.fp8 import FP8Linear

torch.manual_seed(734)
result = {'torch': torch.__version__, 'hip': torch.version.hip,
          'gpu': torch.cuda.get_device_name(0), 'cases': []}
with torch.inference_mode():
    for m, k, n in [(64, 2048, 6144), (1024, 2048, 6144), (4096, 2048, 2048)]:
        case = {'shape': [m, k, n]}
        try:
            layer = torch.nn.Linear(k, n, bias=False, device='cuda', dtype=torch.bfloat16)
            fp8 = FP8Linear(layer)
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
            reference = layer(x)
            actual = fp8(x)
            torch.cuda.synchronize()
            case['relative_mean_error'] = ((reference.float() - actual.float()).abs().mean() / reference.float().abs().mean()).item()
            case['finite'] = bool(torch.isfinite(actual).all())
            for name, op in [('bf16', layer), ('fp8', fp8)]:
                for _ in range(3):
                    op(x)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(10):
                    op(x)
                torch.cuda.synchronize()
                case[name + '_ms'] = (time.perf_counter() - started) * 100
        except Exception as exc:
            case['error'] = type(exc).__name__ + ': ' + str(exc)
        result['cases'].append(case)

print(json.dumps(result, indent=2), flush=True)
