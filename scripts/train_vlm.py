"""The repo's VLM path, made to run.

BLIP cannot be fetched: huggingface.co is refused at the egress proxy from both
the cloud container and the user's machine, and no BLIP weights are cached
anywhere locally. Neither can pretrained GPT-2 text decoding be used -- the
local gpt2/ folder carries config.json and model.safetensors but no vocab.json
or merges.txt, so its BPE cannot be reconstructed.

What DOES exist locally is the trained OpenDriveFM backbone and its 384-d BEV
latent for all 404 keyframes. So the VLM is built the same way the VLA is, the
LLaVA way: a frozen vision encoder, a trainable projector, and a language
decoder -- here trained from scratch over a word-level vocabulary induced from
the caption corpus, because that is what can honestly be built from what is
reachable.

The captions are not free text. They are generated deterministically from the
nuScenes annotations, so every claim in a caption is checkable and the model can
be SCORED rather than admired: nearest-object class, its bearing, its range, and
the object count are each read back out of the generated sentence and compared
against the labels. A captioner is only interesting if you can say how often it
is wrong.
"""
import json, time, numpy as np, torch, torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)

LAB = {r["token"]: r for r in json.load(
    open('/mnt/user-data/uploads/Projects/opendrivefm/outputs/artifacts/scene_labels.json'))}
D = np.load('all_frames.npz', allow_pickle=True)
TOK = [str(t) for t in D['tokens']]
Z = np.load('z_all404.npy')
assert len(TOK) == len(Z) == 404
missing = [t for t in TOK if t not in LAB]
print(f"latents {Z.shape} | labels {len(LAB)} | unmatched {len(missing)}")

# ---------- caption corpus, generated from the labels ----------
VEH = {"vehicle"}; PED = {"human"}
def rbin(r):  return int(round(r / 5.0) * 5)
def cbin(n):  return min(n, 40)
def caption(row):
    o = row["objects"]
    veh = sum(1 for x in o if x["grp"] in VEH)
    ped = sum(1 for x in o if x["grp"] in PED)
    if not o:
        return ["scene", "clear", "no", "objects", "in", "range", "."]
    n = o[0]
    return ["scene", "with", str(cbin(len(o))), "objects", ".",
            "nearest", n["cat"].lower().replace(" ", "_"), "at", str(rbin(n["r"])), "metres",
            n["b"], ".", str(cbin(veh)), "vehicles", str(cbin(ped)), "pedestrians", "."]

caps = [caption(LAB[t]) for t in TOK]
words = sorted({w for c in caps for w in c})
BOS, EOS, PAD = len(words), len(words) + 1, len(words) + 2
V = len(words) + 3
W2I = {w: i for i, w in enumerate(words)}
I2W = {i: w for w, i in W2I.items()}
L = max(len(c) for c in caps) + 2
print(f"vocab {V} words | caption length {L} | e.g. {' '.join(caps[0])}")
seq = np.full((404, L), PAD, np.int64)
for i, c in enumerate(caps):
    ids = [BOS] + [W2I[w] for w in c] + [EOS]
    seq[i, :len(ids)] = ids
seq = torch.from_numpy(seq)

idx = np.random.permutation(404); tr, va = idx[:324], idx[324:]
Zt = torch.from_numpy(Z).float()

# ---------- projector + decoder ----------
E, K = 256, 4
proj = nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Linear(512, K * E))
cfg = GPT2Config(vocab_size=V, n_positions=L + K + 2, n_ctx=L + K + 2,
                 n_embd=E, n_layer=4, n_head=4, bos_token_id=BOS, eos_token_id=EOS)
lm = GPT2LMHeadModel(cfg)
emb = lm.get_input_embeddings()
par = list(proj.parameters()) + list(lm.parameters())
opt = torch.optim.AdamW(par, lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=900)
print(f"projector {sum(p.numel() for p in proj.parameters())/1e6:.2f}M + "
      f"decoder {sum(p.numel() for p in lm.parameters())/1e6:.2f}M trainable")

def loss_on(ids):
    z = Zt[ids]; pre = proj(z).view(len(ids), K, E); t = seq[ids]
    out = lm(inputs_embeds=torch.cat([pre, emb(t)], 1)).logits[:, K:-1]
    return nn.functional.cross_entropy(out.reshape(-1, V), t[:, 1:].reshape(-1),
                                       ignore_index=PAD)

CURVE = []; t0 = time.time()
for st in range(900):
    lm.train(); b = np.random.choice(tr, 16, replace=False)
    opt.zero_grad(); l = loss_on(b); l.backward()
    torch.nn.utils.clip_grad_norm_(par, 1.0); opt.step(); sched.step()
    if st % 10 == 0 or st == 899:
        lm.eval()
        with torch.no_grad(): vl = loss_on(va).item()
        CURVE.append([st, round(l.item(), 4), round(vl, 4)])
        if st % 150 == 0: print(f"  step {st:3d}  train {l.item():.3f}  val {vl:.3f}")
print(f"trained in {time.time()-t0:.0f}s")

@torch.no_grad()
def decode(ids):
    lm.eval(); pre = proj(Zt[ids]).view(len(ids), K, E)
    cur = torch.full((len(ids), 1), BOS, dtype=torch.long)
    for _ in range(L):
        lg = lm(inputs_embeds=torch.cat([pre, emb(cur)], 1)).logits[:, -1]
        cur = torch.cat([cur, lg.argmax(-1, keepdim=True)], 1)
    out = []
    for r in cur[:, 1:].tolist():
        w = []
        for i in r:
            if i in (EOS, PAD): break
            w.append(I2W.get(i, "?"))
        out.append(w)
    return out

# ---------- read the claims back out and score them ----------
def parse(w):
    d = {"n": None, "cat": None, "r": None, "b": None}
    try:
        if "objects" in w and w.index("objects") >= 1:
            d["n"] = int(w[w.index("objects") - 1])
    except Exception: pass
    if "nearest" in w:
        i = w.index("nearest")
        try:
            d["cat"] = w[i + 1]; d["r"] = int(w[i + 3]); d["b"] = w[i + 5]
        except Exception: pass
    return d

pred = decode(va); gold = [caps[i] for i in va]
P = [parse(p) for p in pred]; G = [parse(g) for g in gold]
def acc(k): 
    ok = [1 for p, g in zip(P, G) if g[k] is not None and p[k] == g[k]]
    tot = sum(1 for g in G if g[k] is not None)
    return round(len(ok) / max(1, tot), 4), tot
def mae(k):
    v = [abs(p[k] - g[k]) for p, g in zip(P, G) if g[k] is not None and p[k] is not None]
    return round(float(np.mean(v)), 3) if v else None

# baseline: the majority caption, i.e. what you get with no image at all
from collections import Counter
maj = Counter(tuple(caps[i]) for i in tr).most_common(1)[0][0]
Pb = [parse(list(maj))] * len(va)
def acc_b(k):
    ok = [1 for p, g in zip(Pb, G) if g[k] is not None and p[k] == g[k]]
    return round(len(ok) / max(1, sum(1 for g in G if g[k] is not None)), 4)
def mae_b(k):
    v = [abs(p[k] - g[k]) for p, g in zip(Pb, G) if g[k] is not None and p[k] is not None]
    return round(float(np.mean(v)), 3) if v else None

a_cat, n_cat = acc("cat"); a_b, n_b = acc("b")
res = {
  "nearest_object_class": {"vlm": a_cat, "majority_caption": acc_b("cat"), "n": n_cat},
  "nearest_object_bearing": {"vlm": a_b, "majority_caption": acc_b("b"), "n": n_b},
  "nearest_object_range_mae_m": {"vlm": mae("r"), "majority_caption": mae_b("r")},
  "object_count_mae": {"vlm": mae("n"), "majority_caption": mae_b("n")},
}
print(json.dumps(res, indent=1))
ex = [{"scene": LAB[TOK[va[i]]]["scene"], "token": TOK[va[i]],
       "generated": " ".join(pred[i]), "reference": " ".join(gold[i])} for i in range(6)]
json.dump({
  "status": "RUNS — trained locally, no network",
  "model": f"frozen OpenDriveFM backbone -> z(384) -> trainable projector -> {K} prefix embeddings "
           f"-> 4-layer transformer decoder ({E} embd, vocab {V}) trained from scratch",
  "trainable_params_m": round(sum(p.numel() for p in par) / 1e6, 3),
  "train_samples": len(tr), "val_samples": len(va), "steps": 900, "vocab": V,
  "corpus": "Captions are generated deterministically from the nuScenes annotations, so every "
            "claim in a sentence is checkable. The model never sees the annotations -- only the "
            "384-d BEV latent from the six camera images.",
  "why_not_blip": "huggingface.co returns 403 at the egress proxy from BOTH the cloud container "
                  "and the user's machine, and nothing is cached locally. Salesforce's own weight "
                  "host and download.pytorch.org are refused too. The local gpt2/ folder has "
                  "weights but no vocab.json or merges.txt, so its BPE cannot be reconstructed "
                  "either. This is what can be built from what is reachable, and it is scored "
                  "rather than described.",
  "results": res, "curve": CURVE, "examples": ex,
  "caveats": [
    "The vocabulary is word-level and induced from the caption corpus, so this model can only say "
    "things the corpus can say. It is not a general-purpose captioner and is not comparable to BLIP.",
    "404 keyframes from 10 scenes. The majority-caption row is the number that matters: it is what "
    "you score by ignoring the image entirely.",
    "Captions describe annotated objects with LiDAR returns within 50 m, ordered by range."]},
  open('vlm_local_report.json', 'w'), indent=2)
print("\nwrote vlm_local_report.json")
