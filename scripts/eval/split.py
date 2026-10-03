#!/usr/bin/env python3
"""The dev/test split, as one object that every script reads.

WHY THIS FILE EXISTS (A45)
--------------------------
Line 16 of docs/METHOD_FREEZE.md is the project's central methodological claim:
"Unlimited iteration on dev. The test split is opened once, with the method
already frozen." An audit on 3 October found that claim was not enforced
anywhere in code.

The split itself was real. `outputs/artifacts/val_dev_test.json` was written on
11 September with 75 dev and 75 test scenes from seed 20261116, before the
observability line of work produced any result. But it was never committed, and
no script written after it ever read it. Instead each one invented its own:

    selective.py                dev = scenes where index % 2 == 0
    recal_confound.py           seeded permutation, seed 9
    temporal_observability.py   seed 44
    missed_detection.py         seed 5
    freespace_hump.py           seed 3
    safety_envelope.py          no split at all
    h2_vs_maskcamera.py         no split at all, all 150 scenes

Four different splits and two results with none, including H2, the claim the
paper turns on. A protocol that exists in a document and not in code is not a
protocol.

THE SHA DOES NOT VERIFY, AND THAT IS RECORDED RATHER THAN HIDDEN
----------------------------------------------------------------
METHOD_FREEZE records `sha 3e00ea450bb17507`. That digest could not be
reproduced from the split file by any construction tried: sha256, md5 and
blake2b-8 over the raw file, over the compact and sorted JSON, over the scene
lists in original and sorted order, and over the seed. It may have been
computed over an artifact that no longer exists.

So the original sha is UNVERIFIABLE. What is verifiable is below: DIGEST is
computed by this file from the committed scene lists, and `--check` recomputes
it. The split's provenance rests on the file's 11 September timestamp and its
seed, not on the sha in the freeze file, and that distinction is stated in A45.
"""
from __future__ import annotations
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PATH = os.path.join(ROOT, "outputs/artifacts/val_dev_test.json")

# Recomputed by --check. If this constant and the file ever disagree, the split
# has been edited and every number measured under it is void.
DIGEST = "924a44db544d4818"


def _load():
    with open(PATH) as fh:
        d = json.load(fh)
    dev, test = sorted(d["dev"]), sorted(d["test"])
    if set(dev) & set(test):
        raise SystemExit("split is not disjoint")
    return dev, test, d.get("seed")


def digest():
    dev, test, seed = _load()
    payload = f"{seed}|{','.join(dev)}|{','.join(test)}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def dev_scenes():
    return set(_load()[0])


def test_scenes():
    return set(_load()[1])


def select(scenes, which):
    """Filter an iterable of scene names. which in {dev, test, all}."""
    if which == "all":
        return set(scenes)
    keep = dev_scenes() if which == "dev" else test_scenes()
    return {s for s in scenes if s in keep}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dev, test, seed = _load()
    d = digest()
    print(f"dev {len(dev)} scenes | test {len(test)} scenes | seed {seed}")
    print(f"digest {d}")
    if a.check:
        if d != DIGEST:
            raise SystemExit(f"SPLIT CHANGED: expected {DIGEST}, got {d}")
        print("digest matches the constant in this file")
