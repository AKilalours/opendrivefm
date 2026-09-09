"""Run the repository's own VLM path -- scripts/vlm_scene_understanding.py,
Salesforce BLIP -- over the frames the console displays.

The weights were fetched once from huggingface.co on the user's macOS session
(the desktop workspace and this container are both behind an egress proxy that
refuses it) and are read from disk here. Nothing about the model is simulated:
this is BLIP's ViT encoder cross-attending into its language decoder, generating
token by token, on the real camera pixels the console shows.
"""
import base64, io, json, os, time
import numpy as np
from PIL import Image
os.environ["BLIP_PATH"] = os.path.abspath("blip-base")
import sys; sys.path.insert(0, ".")
from vlm_scene_understanding import caption_scene, load_vlm

BUNDLE = json.load(open('/home/claude/outputs/console/bundle.json'))
CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

def decode(b64):
    raw = base64.b64decode(b64.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))[:, :, ::-1]  # BGR

t0 = time.time(); model, proc, dev = load_vlm()
load_s = time.time() - t0
nparam = sum(p.numel() for p in model.parameters())
print(f"BLIP loaded on {dev} in {load_s:.1f}s | {nparam/1e6:.1f}M params")

out, lat = [], []
for i, f in enumerate(BUNDLE["frames"]):
    caps = {}
    for c in CAMS:
        img = decode(f["cameras"][c]["plain"])
        t = time.time(); caps[c] = caption_scene(img); lat.append(time.time() - t)
    out.append({"scene": f["scene"], "token": f["token"], "captions": caps})
    print(f"  [{i+1}/{len(BUNDLE['frames'])}] {f['scene']}  front: {caps['CAM_FRONT']}", flush=True)

lat = np.array(lat) * 1000
json.dump({
  "status": "RUNS — real BLIP weights, loaded from disk",
  "model": "Salesforce/blip-image-captioning-base (BLIP: ViT-B/16 image encoder → cross-attention "
           f"→ BERT-style text decoder), {nparam/1e6:.1f}M parameters",
  "entrypoint": "scripts/vlm_scene_understanding.py :: caption_scene() — the repository's own function",
  "device": dev, "load_seconds": round(load_s, 1),
  "frames": len(out), "cameras_per_frame": len(CAMS), "captions": len(lat),
  "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1),
                 "p95": round(float(np.percentile(lat, 95)), 1),
                 "mean": round(float(lat.mean()), 1)},
  "how_the_weights_got_here":
      "huggingface.co is refused at the egress proxy from this container AND from the desktop "
      "workspace VM. The weights were downloaded once in a macOS terminal, which is not behind that "
      "proxy, and are read from a local directory via BLIP_PATH. The module was changed to accept a "
      "local path so the same code runs online or offline.",
  "what_it_is_not":
      "BLIP is a generic web-image captioner. It was never trained on driving scenes and has no "
      "notion of range, ego frame or safety. Its captions are included because they are what the "
      "repository's VLM path actually produces — compare them against the scene-description model "
      "trained on this dataset, which is scored rather than read.",
  "results": out,
}, open('vlm_blip_report.json', 'w'), indent=2)
print(f"\nwrote vlm_blip_report.json | {len(lat)} captions | p50 {np.percentile(lat,50):.0f} ms")
