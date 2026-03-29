"""
run_real_demo.py — Find a real nuScenes sample and launch the demo with it.
Run this FIRST to find your actual nuScenes camera images.

Copy to ~/opendrivefm/
Run: python run_real_demo.py
"""
import json, subprocess, sys
from pathlib import Path

# Find a real nuScenes CAM_FRONT image from your manifest
manifest = Path("artifacts/nuscenes_mini_manifest.jsonl")
rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]

# Pick sample from a val scene so it's not one the model trained on
val_scenes = {"scene-0655", "scene-1077"}
val_rows = [r for r in rows
            if r.get("scene", r.get("scene_name","")) in val_scenes]

print(f"Found {len(val_rows)} validation samples")
print("\nReal nuScenes CAM_FRONT images available:")
for i, r in enumerate(val_rows[:5]):
    p = Path(r["cams"]["CAM_FRONT"])
    exists = "✓" if p.exists() else "✗ MISSING"
    print(f"  [{i}] {exists}  {p}")

# Find first existing image
chosen = None
for r in val_rows:
    p = Path(r["cams"]["CAM_FRONT"])
    if p.exists():
        chosen = p
        chosen_row = r
        break

if chosen is None:
    print("\nERROR: No nuScenes images found. Check data/nuscenes/ exists.")
    sys.exit(1)

print(f"\n✓ Using: {chosen}")
print(f"  Scene: {chosen_row.get('scene', chosen_row.get('scene_name','?'))}")
print(f"  Token: {chosen_row['sample_token'][:20]}...")

# Also list all 6 cameras for this sample
print("\nAll 6 cameras for this sample:")
for cam, path in chosen_row["cams"].items():
    p = Path(path)
    exists = "✓" if p.exists() else "✗"
    print(f"  {exists} {cam:25s} {p.name}")

print(f"\nLaunching live demo with real nuScenes image...")
print("Press 0-5 to inject faults, Q to quit\n")

subprocess.run([sys.executable, "live_demo_webcam.py",
                "--image", str(chosen)])
