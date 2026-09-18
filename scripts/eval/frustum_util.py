"""Which voxels any camera could see at all, ignoring occlusion.

Why this exists
---------------
The zero-observability bin is heterogeneous, and that heterogeneity is the
measure's central weakness (A13). Two populations land in it:

  (a) OUT OF VIEW  -- no camera frustum contains the voxel. Nothing was ever
      going to see it, and the model predicts it from context alone.
  (b) OCCLUDED     -- in view, but behind the first surface along every ray.
      Car interiors, wall backs. The model often predicts these WELL, by
      continuation rather than by sight.

Pooled, they behave oppositely to the visible deciles and drag the zero bin to
a better calibration than decile 1, which is what made H1 fail on all voxels
and what fragmented the recalibration groups.

Splitting them is a change to how results are REPORTED, not a change to the
measure. Observability itself is untouched: no new parameter, no new
derivation. Cost is one projection per camera, ~0.05 s/frame, so it is
recomputed rather than stored.
"""
from __future__ import annotations

import numpy as np

OCC_N, OCC_RES, OCC_RNG = 200, 0.4, 40.0
OCC_NZ, OCC_Z0 = 16, -1.0
CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

_VOX = None


def voxel_centres():
    global _VOX
    if _VOX is None:
        ax = (np.arange(OCC_N) + 0.5) * OCC_RES - OCC_RNG
        az = (np.arange(OCC_NZ) + 0.5) * OCC_RES + OCC_Z0
        X, Y, Z = np.meshgrid(ax, ax, az, indexing="ij")
        _VOX = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    return _VOX


def in_any_frustum(cams, quat_to_rot):
    """True where at least one camera's image contains the voxel centre."""
    vox = voxel_centres()
    out = np.zeros(vox.shape[0], bool)
    for c in CAMS:
        cam = cams.get(c)
        if cam is None:
            continue
        W = cam.get("width", 1600)
        H = cam.get("height", 900)
        R = quat_to_rot(cam["sensor2ego_rotation"])
        t = np.asarray(cam["sensor2ego_translation"], np.float32)
        K = np.asarray(cam.get("cam_intrinsic", cam.get("intrinsic")),
                       np.float32)
        p = (vox - t) @ R
        z = p[:, 2]
        good = z > 0.1
        zz = np.where(good, z, 1.0)
        u = K[0, 0] * p[:, 0] / zz + K[0, 2]
        v = K[1, 1] * p[:, 1] / zz + K[1, 2]
        out |= good & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return out.reshape(OCC_N, OCC_N, OCC_NZ)
