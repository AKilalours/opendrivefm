#!/usr/bin/env python3
"""Benchmark the traced model on whatever accelerator this machine actually has.

Run this on a machine with a GPU and the console grows a measured accelerator
row. On Apple Silicon that is MPS; on an NVIDIA box, CUDA. It refuses to invent
a number: with no accelerator it says so and writes nothing.

    python scripts/bench_device.py

Writes outputs/artifacts/device_report.json. Rebuild the console afterwards:

    python scripts/export_console_bundle.py && python scripts/build_console.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "outputs/artifacts/opendrivefm_v11.pt"
OUT = ROOT / "outputs/artifacts/device_report.json"
ITERS, WARMUP, BLOCKS = 60, 10, 3


def pick():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (Metal)"
    return None, None


def p50(f, n, warm, sync):
    for _ in range(warm):
        f()
    sync()
    t = []
    for _ in range(n):
        a = time.perf_counter(); f(); sync(); t.append(time.perf_counter() - a)
    return float(np.percentile(np.array(t) * 1000, 50))


def main():
    dev, name = pick()
    if dev is None:
        print("No accelerator on this machine (no CUDA, no MPS). Nothing written -- "
              "a latency figure for hardware that is not here would be a fabrication.")
        return
    if not TS.exists():
        raise SystemExit(f"traced module not found at {TS}; run scripts/export_torchscript.py first")

    m = torch.jit.load(str(TS), map_location="cpu").eval()
    torch.manual_seed(0)
    x = torch.rand(1, 6, 4, 3, 90, 160)   # four-frame window: the validated configuration
    v = torch.zeros(1, 2)

    with torch.no_grad():
        cpu_out = m(x, v)
        cpu_ms = p50(lambda: m(x, v), ITERS, WARMUP, lambda: None)

        md = torch.jit.load(str(TS), map_location=dev).eval()
        xd, vd = x.to(dev), v.to(dev)
        sync = (torch.cuda.synchronize if dev == "cuda"
                else getattr(torch, "mps", None) and torch.mps.synchronize or (lambda: None))
        try:
            dev_out = md(xd, vd)
        except NotImplementedError as e:
            if "_transformer_encoder_layer_fwd" in str(e):
                raise SystemExit(
                    "\nThis module was traced with nn.TransformerEncoderLayer's fused fast path,\n"
                    "which has no MPS kernel. The graph, not your machine, is the problem.\n\n"
                    "Re-export it -- scripts/export_torchscript.py now disables the fast path --\n"
                    "then run this again:\n\n"
                    "    python scripts/export_torchscript.py\n"
                    "    python scripts/bench_device.py\n\n"
                    "Do not set PYTORCH_ENABLE_MPS_FALLBACK=1 to get past this: it silently runs\n"
                    "the attention layers on CPU, and the number you would get back would be a\n"
                    "mixed-device figure reported as a GPU one.\n")
            raise
        # Three separate measurement blocks. Metal compiles shaders lazily, so a
        # first run reads slower than a second; one block would report whichever
        # of those the machine happened to be in.
        dev_blocks = [p50(lambda: md(xd, vd), ITERS, WARMUP, sync) for _ in range(BLOCKS)]
        dev_ms = float(np.median(dev_blocks))

    def flat(o):
        if torch.is_tensor(o): return [o.detach().float().cpu().reshape(-1)]
        if isinstance(o, dict): return sum((flat(v) for v in o.values()), [])
        if isinstance(o, (list, tuple)): return sum((flat(v) for v in o), [])
        return []
    a, b = torch.cat(flat(cpu_out)), torch.cat(flat(dev_out))
    diff = float((a - b).abs().max())

    rep = {
        "status": "MEASURED",
        "device": f"{dev} — {name}",
        "torch": torch.__version__,
        "module": "outputs/artifacts/opendrivefm_v11.pt (TorchScript, traced + frozen)",
        "iterations": ITERS, "warmup": WARMUP,
        "cpu_p50_ms": round(cpu_ms, 2),
        "forward_p50_ms": round(dev_ms, 2),
        "forward_p50_blocks_ms": [round(b, 2) for b in dev_blocks],
        "forward_spread_pct": round(100 * (max(dev_blocks) - min(dev_blocks)) / dev_ms, 1),
        "traced_input_shape": str(tuple(x.shape)),
        "speedup_vs_cpu": round(cpu_ms / max(1e-9, dev_ms), 2),
        "max_abs_diff": float(f"{diff:.3g}"),
        "note": ("The same traced module, moved with .to(device) and nothing else. The output "
                 "difference is the accelerator's arithmetic, not a different graph."),
        "caveats": [
            "Batch 1, the shape the runner actually uses. Throughput on larger batches is a "
            "different measurement and is not claimed here.",
            "Includes the host-to-device copy, because a deployed runner pays it too.",
            "Median of three measurement blocks; the spread between them is reported, because "
            "Metal compiles shaders lazily and a single block reports whichever warm-up state the "
            "machine was in.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
