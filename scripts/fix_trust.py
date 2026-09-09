"""Remap the checkpoint's trust-scorer keys onto the current module layout.

The checkpoint stores CameraTrustScorer as one flat Sequential named `cnn`.
The current class splits that same stack into `trunk` (the conv tower), `pool`
(parameter-free) and `cnn_head` (the MLP). No weights changed shape, only their
addresses -- so this is a rename, not a conversion, and every tensor is checked
for shape agreement before it is accepted.

    cnn.0  Conv2d(3, 32, 5)   -> trunk.0
    cnn.1  BatchNorm2d(32)    -> trunk.1
    cnn.3  Conv2d(32, 64, 5)  -> trunk.3
    cnn.4  BatchNorm2d(64)    -> trunk.4
    cnn.8  Linear(64, 16)     -> cnn_head.0
    cnn.10 Linear(16, 1)      -> cnn_head.2

stat_running_mean / stat_calibrated are buffers added after this checkpoint was
written. They stay at their defaults, which means "statistics not calibrated" --
the honest state, and what the calibration script exists to fill in.
"""
import re, torch

MAP = {"cnn.0": "trunk.0", "cnn.1": "trunk.1", "cnn.3": "trunk.3",
       "cnn.4": "trunk.4", "cnn.8": "cnn_head.0", "cnn.10": "cnn_head.2"}

def remap_trust_keys(sd, model_sd, prefix="backbone.trust_scorer."):
    out, renamed, rejected = dict(sd), [], []
    for k in list(sd):
        if prefix not in k: continue
        tail = k.split(prefix, 1)[1]
        m = re.match(r"(cnn\.\d+)\.(.+)", tail)
        if not m or m.group(1) not in MAP: continue
        new = k.replace(prefix + m.group(1), prefix + MAP[m.group(1)])
        tgt = new.split("model.", 1)[-1] if new.startswith("model.") else new
        if tgt in model_sd and model_sd[tgt].shape == sd[k].shape:
            out[new] = sd[k]; out.pop(k); renamed.append((tail, MAP[m.group(1)] + "." + m.group(2)))
        else:
            rejected.append((tail, tuple(sd[k].shape),
                             tuple(model_sd[tgt].shape) if tgt in model_sd else None))
    return out, renamed, rejected
