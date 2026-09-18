"""Vision-language reasoning over the driving scene, with the safety-critical
token scored rather than read.

A captioner that says "a street with cars parked on it" is not a perception
component. What an autonomy stack needs from language is a claim it can be held
to, so this model emits a structured narrative whose last clause is a hazard
decision -- is there a vulnerable road user inside 20 m in the forward arc --
and that clause is parsed back out and scored with precision, recall and a
confidence read from the decoder's own softmax.

Same LLaVA shape as the rest of the repo: frozen OpenDriveFM backbone -> 384-d
BEV latent -> trainable projector -> prefix embeddings -> transformer decoder.
The model never sees an annotation; it sees the six camera images through the
frozen encoder and nothing else.
"""
import json, time, numpy as np, torch, torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
# --- split / seed / paths: configured by environment, never hardcoded --------
import os as _os, sys as _sys
_ODFM_ROOT  = _os.environ.get("ODFM_ROOT", _os.path.dirname(_os.path.abspath(__file__)))
_ODFM_SPLIT = _os.environ.get("ODFM_SPLIT", "standard")
_ODFM_SEED  = int(_os.environ.get("ODFM_SEED", "0"))
_LABELS     = _os.environ.get(
    "ODFM_LABELS",
    _os.path.join(_ODFM_ROOT, "outputs", "artifacts", "scene_labels.json"))
if not _os.path.exists(_LABELS):
    _alt = _os.path.join(_ODFM_ROOT, "artifacts", "scene_labels.json")
    if _os.path.exists(_alt): _LABELS = _alt
for _p in (_os.path.join(_ODFM_ROOT, "src"), _ODFM_ROOT):
    if _p not in _sys.path: _sys.path.insert(0, _p)
try:
    from opendrivefm.data import splits as _splits
except ImportError:
    import splits as _splits
# -----------------------------------------------------------------------------
torch.manual_seed(_ODFM_SEED); np.random.seed(_ODFM_SEED); torch.set_num_threads(2)

LAB = {r["token"]: r for r in json.load(
    open(_LABELS))}
D = np.load(_os.environ.get('ODFM_FEATURES_DIR', _ODFM_ROOT) + '/all_frames.npz', allow_pickle=True)
TOK = [str(t) for t in D['tokens']]; MO = D['motion']
Z = np.load(_os.environ.get('ODFM_FEATURES_DIR', _ODFM_ROOT) + '/z_all404.npy')

FWD = {"ahead", "front-left", "front-right"}
VRU = {"human", "vehicle.bicycle", "vehicle.motorcycle"}
def is_vru(o): return o["grp"] == "human" or o["cat"].lower() in ("bicycle", "motorcycle")
def rbin(r): return int(round(r / 5.0) * 5)
def cbin(n): return min(n, 40)

def hazard(row):
    """A VRU inside 20 m in the forward arc. This is the clause that matters."""
    v = [o for o in row["objects"] if is_vru(o) and o["b"] in FWD and o["r"] <= 20.0]
    return (True, min(v, key=lambda o: o["r"])) if v else (False, None)

def caption(row, spd):
    o = row["objects"]
    veh = sum(1 for x in o if x["grp"] == "vehicle")
    ped = sum(1 for x in o if x["grp"] == "human")
    haz, n = hazard(row)
    sp = "stopped" if spd < 0.5 else "slow" if spd < 5 else "moderate" if spd < 10 else "fast"
    head = (["scene", "clear"] if not o else
            ["scene", str(cbin(len(o))), "objects", "nearest",
             o[0]["cat"].lower().replace(" ", "_"), str(rbin(o[0]["r"])), "metres", o[0]["b"]])
    return head + [".", "ego", sp, ".", str(cbin(veh)), "vehicles", str(cbin(ped)), "pedestrians", "."] + \
        (["hazard", "high", n["cat"].lower().replace(" ", "_"), str(rbin(n["r"])), "metres", n["b"], "."]
         if haz else ["hazard", "low", "."])

spd = np.linalg.norm(MO[:, 1:3], axis=1) * (MO[:, 0] > 0)
caps = [caption(LAB[t], spd[i]) for i, t in enumerate(TOK)]
haz_y = np.array([hazard(LAB[t])[0] for t in TOK], bool)
print(f"hazard positives {haz_y.sum()}/{len(haz_y)} = {haz_y.mean():.3f} base rate")

words = sorted({w for c in caps for w in c})
BOS, EOS, PAD = len(words), len(words) + 1, len(words) + 2
V = len(words) + 3; W2I = {w: i for i, w in enumerate(words)}; I2W = {i: w for w, i in W2I.items()}
L = max(len(c) for c in caps) + 2
seq = np.full((len(caps), L), PAD, np.int64)
for i, c in enumerate(caps): 
    ids = [BOS] + [W2I[w] for w in c] + [EOS]; seq[i, :len(ids)] = ids
seq = torch.from_numpy(seq)
print(f"vocab {V} | length {L} | e.g. {' '.join(caps[int(np.argmax(haz_y))])}")

# Scene-level split. A random keyframe permutation put keyframes 0.5 s
# apart on both sides of the boundary, so val scored recall, not
# generalisation. Whole scenes are held out instead.
tr, va = _splits.split_indices(TOK, split=_ODFM_SPLIT, labels_path=_LABELS)
print(_splits.describe(_ODFM_SPLIT, _LABELS))
print(f'train {len(tr)} | val {len(va)} keyframes, disjoint by scene')
Zt = torch.from_numpy(Z).float()
E, K = 256, 4
proj = nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Linear(512, K * E))
lm = GPT2LMHeadModel(GPT2Config(vocab_size=V, n_positions=L + K + 2, n_ctx=L + K + 2,
                                n_embd=E, n_layer=4, n_head=4, bos_token_id=BOS, eos_token_id=EOS))
emb = lm.get_input_embeddings(); par = list(proj.parameters()) + list(lm.parameters())
opt = torch.optim.AdamW(par, lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1100)
print(f"{sum(p.numel() for p in par)/1e6:.2f}M trainable")

def loss_on(ids):
    pre = proj(Zt[ids]).view(len(ids), K, E); t = seq[ids]
    out = lm(inputs_embeds=torch.cat([pre, emb(t)], 1)).logits[:, K:-1]
    return nn.functional.cross_entropy(out.reshape(-1, V), t[:, 1:].reshape(-1), ignore_index=PAD)

CURVE = []; t0 = time.time()
for st in range(1100):
    lm.train(); b = np.random.choice(tr, 16, replace=False)
    opt.zero_grad(); l = loss_on(b); l.backward()
    torch.nn.utils.clip_grad_norm_(par, 1.0); opt.step(); sched.step()
    if st % 10 == 0 or st == 1099:
        lm.eval()
        with torch.no_grad(): vl = loss_on(va).item()
        CURVE.append([st, round(l.item(), 4), round(vl, 4)])
        if st % 200 == 0: print(f"  step {st:4d} train {l.item():.3f} val {vl:.3f}", flush=True)
print(f"trained in {time.time()-t0:.0f}s")

HI, LO = W2I["high"], W2I["low"]
@torch.no_grad()
def decode(ids):
    lm.eval(); pre = proj(Zt[ids]).view(len(ids), K, E)
    cur = torch.full((len(ids), 1), BOS, dtype=torch.long)
    phi = np.zeros(len(ids))
    for step in range(L):
        lg = lm(inputs_embeds=torch.cat([pre, emb(cur)], 1)).logits[:, -1]
        nxt = lg.argmax(-1, keepdim=True)
        # confidence on the hazard token: softmax over {high, low} the moment
        # the model has just emitted "hazard"
        prev = cur[:, -1]
        at = (prev == W2I["hazard"]).numpy()
        if at.any():
            p = torch.softmax(lg[:, [HI, LO]], -1)[:, 0].numpy()
            phi = np.where(at, p, phi)
        cur = torch.cat([cur, nxt], 1)
    outs = []
    for r in cur[:, 1:].tolist():
        w = []
        for i in r:
            if i in (EOS, PAD): break
            w.append(I2W.get(i, "?"))
        outs.append(w)
    return outs, phi

pred, conf = decode(va)
def hz(w): return ("high" in w and w.index("high") > 0 and w[w.index("high") - 1] == "hazard")
yh = np.array([hz(p) for p in pred]); yt = haz_y[va]
tp = int((yh & yt).sum()); fp = int((yh & ~yt).sum()); fn = int((~yh & yt).sum()); tn = int((~yh & ~yt).sum())
prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn); f1 = 2 * prec * rec / max(1e-9, prec + rec)
def auroc(s, y):
    """Rank-based, ties averaged. The earlier version was arithmetically wrong
    and produced 0.046, which is not a value AUROC can take for a sane score."""
    P, N = int(y.sum()), int((~y).sum())
    if P == 0 or N == 0: return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties, which matters here: argmax decoding saturates
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - P * (P + 1) / 2) / (P * N))
au = auroc(conf, yt)
base = float(yt.mean())
res = {"hazard_precision": round(prec, 4), "hazard_recall": round(rec, 4), "hazard_f1": round(f1, 4),
       "hazard_auroc": None if au is None else round(au, 4),
       "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
       "confidence_saturated_frac": round(float((conf >= 0.999).mean()), 3),
       "positives_in_val": int(yt.sum()), "val": int(len(yt)), "base_rate": round(base, 4),
       "always_high_f1": round(2 * base / (1 + base), 4)}
print(json.dumps(res, indent=1))
ex = [{"scene": LAB[TOK[va[i]]]["scene"], "token": TOK[va[i]], "generated": " ".join(pred[i]),
       "reference": " ".join(caps[va[i]]), "hazard_confidence": round(float(conf[i]), 3),
       "correct": bool(hz(pred[i]) == yt[i])} for i in np.argsort(-conf)[:3].tolist() + np.argsort(conf)[:3].tolist()]
json.dump({
  'split': _ODFM_SPLIT,
  'split_scenes': {k: list(v) for k, v in _splits.SPLITS[_ODFM_SPLIT].items()},
  'seed': _ODFM_SEED,
  'split_method': 'scene-level holdout; no scene appears in both train and val',
 "status": "RUNS — trained here",
 "model": f"frozen OpenDriveFM backbone -> z(384) -> trainable projector -> {K} prefix embeddings -> "
          f"4-layer transformer decoder ({E} embd, vocab {V}), trained from scratch",
 "trainable_params_m": round(sum(p.numel() for p in par) / 1e6, 3),
 "train_samples": len(tr), "val_samples": len(va), "steps": 1100, "vocab": V,
 "what": "The narrative ends in a hazard clause -- a vulnerable road user inside 20 m in the forward "
         "arc -- so the sentence carries a decision that can be scored instead of admired. Confidence "
         "is the decoder's own softmax over high/low at the moment it emits the token.",
 "results": res, "curve": CURVE, "examples": ex,
 "caveats": ["404 keyframes, 10 scenes. Precision and recall on 80 held-out frames move in large steps.",
             "The hazard rule is geometric (VRU class, forward arc, 20 m) and read from annotations; the "
             "model sees only the camera latent.",
             "A word-level vocabulary induced from the corpus: this model can only say what the corpus "
             "can say, and is not a general-purpose captioner."]},
 open(_os.environ.get('ODFM_OUT', '.') + f'/vllm_report_{_ODFM_SPLIT}_seed{_ODFM_SEED}.json', 'w'), indent=2)
print(f"\nwrote vllm_report_{_ODFM_SPLIT}_seed{_ODFM_SEED}.json")
