#!/usr/bin/env python3
"""Dump a semantic summary of every keyframe, straight from the nuScenes
annotations, for the VLM caption corpus. No rendering, no model -- these are
labels, and they are what the captioner is trained and scored against."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import odfm_overlay as O, odfm_tables as T

OUT = Path(__file__).resolve().parents[1] / "outputs/artifacts/scene_labels.json"
BEARINGS = ["ahead", "front-left", "left", "rear-left", "behind",
            "rear-right", "right", "front-right"]


def bearing(x, y):
    a = np.degrees(np.arctan2(y, x)) % 360          # ego x forward, y left
    return BEARINGS[int(((a + 22.5) % 360) // 45)]


def main():
    tab = T.tables()
    rows = []
    for stok, toks in tab.scene_samples.items():
        for tok in toks:
            objs = []
            for b in tab.boxes_ego(tok):
                if b["num_lidar_pts"] <= 0:
                    continue
                r = float(np.hypot(b["centre"][0], b["centre"][1]))
                if r > 50:
                    continue
                objs.append({
                    "cat": O.display_name(b["category"]),
                    "grp": b["category"].split(".")[0],
                    "r": round(r, 1),
                    "b": bearing(b["centre"][0], b["centre"][1]),
                    "vis": b["visibility_level"],
                    "pts": int(b["num_lidar_pts"]),
                })
            objs.sort(key=lambda o: o["r"])
            rows.append({"token": tok, "scene": tab.scene_name(tok), "objects": objs})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows))
    n = sum(len(r["objects"]) for r in rows)
    print(f"wrote {OUT.name}  {len(rows)} keyframes  {n} objects  {OUT.stat().st_size/1e3:.0f} KB")


if __name__ == "__main__":
    main()
