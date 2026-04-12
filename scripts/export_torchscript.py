"""
export_torchscript.py — Export OpenDriveFM to TorchScript for C++ profiling

Run this first before bench_latency:
    python scripts/export_torchscript.py \
        --ckpt outputs/artifacts/checkpoints_v11_temporal/best_val_ade.ckpt \
        --out  outputs/artifacts/opendrivefm_v11.pt

Then run C++ profiler:
    cd build && ./bench_latency --model ../outputs/artifacts/opendrivefm_v11.pt
"""
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.opendrivefm.models.model import OpenDriveFM


def export(ckpt_path: str, out_path: str, device: str = "cpu"):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Strip Lightning prefix if present
    state = {k.replace("model.", "", 1): v for k, v in state.items()}

    model = OpenDriveFM()
    model.load_state_dict(state, strict=False)
    model.eval()

    # Dummy input matching bench_latency default config
    B, V, T, H, W = 1, 6, 1, 90, 160
    imgs = torch.randn(B * V * T, 3, H, W)
    vel  = torch.randn(B, 2)

    print("Tracing model to TorchScript...")
    with torch.no_grad():
        traced = torch.jit.trace(model, (imgs, vel), strict=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(out_path)
    print(f"Saved TorchScript model: {out_path}")
    print("Now run: cd build && ./bench_latency --model ../outputs/artifacts/opendrivefm_v11.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/artifacts/checkpoints_v11_temporal/best_val_ade.ckpt")
    ap.add_argument("--out",  default="outputs/artifacts/opendrivefm_v11.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    export(args.ckpt, args.out, args.device)
