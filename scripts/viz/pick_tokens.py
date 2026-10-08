#!/usr/bin/env python3
"""Pick one distinct sealed-test token per figure, by a stated rule.

Rule for the pipeline figure (Fig 1): among a fixed sample of test-split
tokens, take the one whose zero-observability fraction is closest to the
sample median -- i.e. a typical frame, not a flattering one. The sample is
drawn with a fixed seed and the choice is written to an artifact so it is
reproducible and auditable.
"""
import glob, importlib.util, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED, NSAMP = 20261108, 150
EXCLUDE = {"dc03fe4e0a374592af7701c66d5276c4"}   # already used by Figs 3/4/5


def main():
    sp = importlib.util.spec_from_file_location(
        "sp", os.path.join(ROOT, "scripts/eval/split.py"))
    S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
    test = set(S.test_scenes())

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    def has_imgs(r):
        for c in r["cams"].values():
            fn = c["filename"]
            jp = fn if os.path.exists(fn) else os.path.join(
                ROOT, "dataset/nuscenes", fn.split("nuscenes/")[-1])
            if not os.path.exists(jp):
                return False
        return True

    # the figure needs all six rig images locally, so the pool is the test
    # tokens whose full rig is present in the local nuScenes mirror
    cand = [r["token"] for r in index
            if r["scene"] in test and r["token"] not in EXCLUDE
            and os.path.exists(os.path.join(ROOT, "data/pack/obs_max",
                                            r["token"] + ".npy"))
            and has_imgs(r)]
    rng = np.random.default_rng(SEED)
    samp = [cand[i] for i in rng.choice(len(cand), min(NSAMP, len(cand)),
                                        replace=False)]
    frac = {}
    for t in samp:
        o = np.load(os.path.join(ROOT, "data/pack/obs_max", t + ".npy"))
        frac[t] = float((o == 0).mean())
    med = float(np.median(list(frac.values())))
    t1 = min(frac, key=lambda t: abs(frac[t] - med))

    out = dict(seed=SEED, n_sampled=len(samp), split="test",
               median_zero_obs_fraction=round(med, 6),
               fig1_token=t1, fig1_zero_obs_fraction=round(frac[t1], 6),
               excluded=sorted(EXCLUDE), rule="closest to sample median zero-obs fraction")
    p = os.path.join(ROOT, "outputs/artifacts/figure_tokens.json")
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
