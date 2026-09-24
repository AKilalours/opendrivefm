"""Regression tests for the observability geometry.

Every test here exists because a real bug got through. Three separate defects
in this line of work were caught by a number looking implausible to a human and
none by a test:

  A30  first verified run taken instead of the longest
  A36  clearance envelope starting inside the road surface
  A34  evaluation scoped so the competing baseline became constant

The first two were the same quantity computed in two files. That duplication is
gone (scripts/eval/corridor.py) and these tests hold the line.

No dataset required: everything is synthetic.
"""
from __future__ import annotations
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


corridor = _load("scripts/eval/corridor.py", "corridor")
B = _load("scripts/eval/build_observability_ray.py", "build_obs_ray")
FREE, RES, Z0, NZ, N = 17, 0.4, -1.0, 16, 200


# --------------------------------------------------------------------------
# A30: the run selection rule
# --------------------------------------------------------------------------

def test_run_beyond_an_obstacle_is_not_credited():
    """A38: clear road on the far side of a wall is not corridor the ego has."""
    ok = np.zeros(120, bool)
    ok[0:20] = True           # 20 cells = 8 m, starting at the bumper
    ok[60:119] = True         # a much longer run, but 24 m further out
    a, n = corridor.longest_run(ok, 0, max_start=int(15.0 / RES))
    assert (a, n) == (0, 20), "credited a run beyond an obstacle"


def test_longest_run_not_first():
    """A short run near the start must not win over a long one further out.

    This is A30 exactly: the first verified cell sits in the near-field band
    where coverage is marginal, so a first-run rule returns ~2 m while the
    corridor is verified for tens of metres beyond it.
    """
    ok = np.zeros(60, bool)
    ok[2:5] = True            # a 3-cell run early
    ok[20:50] = True          # a 30-cell run later
    a, n = corridor.longest_run(ok, 0, max_start=int(15.0 / RES))
    assert (a, n) == (20, 30), "took the first run instead of the longest"


def test_longest_run_empty():
    assert corridor.longest_run(np.zeros(10, bool), 0, 10) == (0, 0)


# --------------------------------------------------------------------------
# A36: the clearance envelope
# --------------------------------------------------------------------------

def test_envelope_starts_above_the_road_surface():
    """clear_from must exclude the road.

    Measured height profile of non-free voxels in the lane: levels k=0,1,2 are
    100% occupied (the road) and k=3 is 54% road bleed. clear_from = 0.4 m puts
    the first level at k=3 and therefore asks the road to be free, which
    returned a near-zero corridor on every frame.
    """
    k0 = int((corridor.CLEAR_FROM - Z0) / RES)
    assert k0 >= 4, (f"envelope starts at level k={k0}, which is inside the "
                     f"road surface; A36 fixed this")


def test_envelope_has_an_upper_bound():
    """Overhead structure is not an obstacle. Without a ceiling, a tree canopy
    or a gantry blocks the corridor."""
    assert corridor.CLEAR_TO < 5.4
    k1 = int((corridor.CLEAR_TO - Z0) / RES)
    assert k1 <= 11


def test_corridor_is_measured_from_the_bumper():
    assert corridor.BUMPER > 0, "corridor must start at the vehicle front"


# --------------------------------------------------------------------------
# The duplication itself
# --------------------------------------------------------------------------

def test_one_corridor_implementation():
    """Both callers must be the SAME function object, not two copies that agree
    today. A30 and A36 were both caused by two copies drifting apart."""
    se = _load("scripts/eval/safety_envelope.py", "safety_envelope")
    rh = _load("scripts/viz/render_hd.py", "render_hd")
    # Compare code objects: importlib gives each loader a fresh module object,
    # so `is` on the function would fail even when both import the same source.
    assert se.verified_free.__code__ is rh.verified_free.__code__
    assert se.verified_free.__code__.co_filename.endswith("corridor.py")


def test_corridor_end_to_end_on_a_synthetic_scene():
    """Free lane with a wall at 20 m: the corridor must stop at the wall."""
    cls = np.full((N, N, NZ), FREE, np.int16)
    cls[:, :, 0:3] = 11                      # road surface, must not block
    obs = np.ones((N, N, NZ), np.float32)
    i_wall = int((40.0 + 20.0) / RES)
    cls[i_wall:i_wall + 2, :, 3:12] = 15     # a wall across the lane
    start, length, _, sf, ss = corridor.verified_free(cls, obs)
    assert start == pytest.approx(0.0, abs=0.5)
    end = start + length
    assert 16.0 < end < 19.0, f"corridor ran to {end} m, wall is at 17.6 m"
    assert sf is not None and sf < 0.8, "should stop on an obstacle"


def test_road_surface_alone_does_not_block():
    """The regression A36 caught: a lane containing only road must be clear."""
    cls = np.full((N, N, NZ), FREE, np.int16)
    cls[:, :, 0:3] = 11
    obs = np.ones((N, N, NZ), np.float32)
    _, length, _, _, _ = corridor.verified_free(cls, obs)
    assert length > 30.0, "the road surface is blocking its own corridor"


def test_unobserved_lane_yields_no_corridor():
    cls = np.full((N, N, NZ), FREE, np.int16)
    obs = np.zeros((N, N, NZ), np.float32)
    _, length, _, _, _ = corridor.verified_free(cls, obs)
    assert length == 0.0, "free but unseen must not count as verified"


# --------------------------------------------------------------------------
# The ray marcher
# --------------------------------------------------------------------------

def _forward_camera(hfov=90.0, W=320, H=180, height=1.5):
    """A pinhole looking along +x, optical convention (+z forward, +y down)."""
    f = W / (2 * np.tan(np.radians(hfov) / 2))
    R = np.stack([np.array([0.0, -1.0, 0.0]),     # image +x = ego -y
                  np.array([0.0, 0.0, -1.0]),     # image +y = ego -z
                  np.array([1.0, 0.0, 0.0])], 1)  # optical axis = ego +x
    w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
         (R[1, 0] - R[0, 1]) / (4 * w)]
    assert np.allclose(B.M.quat_to_rot(q), R, atol=1e-6)
    return dict(sensor2ego_translation=[0.0, 0.0, height],
                sensor2ego_rotation=q,
                cam_intrinsic=[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
                width=W, height=H)


def _vox(x, y, z):
    return ((int((x + 40.0) / RES) * N) + int((y + 40.0) / RES)) * NZ \
        + int((z - Z0) / RES)


def test_marcher_sees_ahead_in_an_empty_scene():
    occ = np.zeros(N * N * NZ, bool)
    cam = _forward_camera()
    vis = B.march_visibility(occ, cam, cam["width"], cam["height"], stride=2)
    assert vis[_vox(10.0, 0.0, 1.5)], "a clear cell 10 m ahead is not visible"


def test_marcher_is_blocked_by_a_wall():
    occ = np.zeros((N, N, NZ), bool)
    i = int((40.0 + 10.0) / RES)
    occ[i:i + 2, :, :] = True                    # full-height wall at 10 m
    cam = _forward_camera()
    vis = B.march_visibility(occ.reshape(-1), cam, cam["width"], cam["height"],
                             stride=2)
    assert vis[_vox(5.0, 0.0, 1.5)], "a cell in front of the wall is hidden"
    assert not vis[_vox(20.0, 0.0, 1.5)], "a cell behind the wall is visible"


def test_marcher_marks_the_wall_itself():
    """First-hit semantics: the blocking surface IS seen, its interior is not."""
    occ = np.zeros((N, N, NZ), bool)
    i = int((40.0 + 10.0) / RES)
    occ[i:i + 6, :, :] = True
    cam = _forward_camera()
    vis = B.march_visibility(occ.reshape(-1), cam, cam["width"], cam["height"],
                             stride=2)
    assert vis[_vox(10.2, 0.0, 1.5)], "the front face of the wall is not seen"
    assert not vis[_vox(12.0, 0.0, 1.5)], "the inside of the wall is seen"


def test_fast_path_is_bit_identical():
    """A35's precomputed table must equal the shipped marcher exactly."""
    mb = _load("scripts/perf/march_bench.py", "march_bench")
    rng = np.random.default_rng(0)
    occ = rng.random((N, N, NZ)) < 0.02
    occ[:, :, 0:2] = True
    cam = _forward_camera()
    slow = B.march_visibility(occ.reshape(-1), cam, cam["width"],
                              cam["height"], stride=4)
    tab, val = mb.build_table(cam, stride=4)
    fast = mb.march_fast(occ.reshape(-1), tab, val)
    assert int((slow != fast).sum()) == 0


# ---------------------------------------------------------------------------
# A41: the export gate that could not catch its own failure.
#
# The first Stage-2 export collapsed each BEV column to one voxel, described
# the road surface and discarded everything standing on it. The gate written to
# catch that asked for ">= 1.0% dynamic classes"; the broken export measured
# 1.2% and PASSED. These tests pin the replacement, which is written against
# the failure mode itself rather than against a class histogram that resembles
# it.
# ---------------------------------------------------------------------------
FREE_CLS = 17


def _column_stats(cls):
    """Mirror of the gate in scripts/pod/mini_gate.py."""
    col = (cls != FREE_CLS).sum(-1)
    nz = col[col > 0]
    if nz.size == 0:
        return 0.0, 1.0
    return float(nz.mean()), float((nz == 1).mean())


def _collapsed_export(n=40, nz=16):
    """What the broken export produced: exactly one occupied voxel per column,
    at the ground, and that voxel is drivable surface."""
    cls = np.full((n, n, nz), FREE_CLS, np.int16)
    cls[:, :, 0] = 11                      # drivable surface, one voxel deep
    return cls


def _healthy_export(n=40, nz=16):
    """Road surface plus things standing on it, which is what a per-voxel
    export looks like: several occupied voxels in most occupied columns."""
    cls = np.full((n, n, nz), FREE_CLS, np.int16)
    cls[:, :, 0:2] = 11                    # road, two voxels deep
    cls[10:20, 10:14, 0:8] = 4             # a car
    cls[30:38, 5:35, 0:12] = 15            # a building
    return cls


def test_gate_rejects_a_column_collapse():
    vpc, single = _column_stats(_collapsed_export())
    assert vpc == pytest.approx(1.0), vpc
    assert single == pytest.approx(1.0), single
    assert not (vpc >= 2.0), "the structural gate must fail on a column collapse"
    assert not (single <= 0.50), "the single-column gate must fail on a collapse"


def test_gate_passes_a_healthy_export():
    vpc, single = _column_stats(_healthy_export())
    assert vpc >= 2.0, vpc
    assert single <= 0.50, single


def test_the_old_dynamic_gate_could_not_have_caught_it():
    """Recorded so the reason the replacement exists cannot be forgotten: the
    collapsed export's dynamic share clears the >= 1.0% threshold it was
    supposed to fail."""
    cls = _collapsed_export()
    cls[0:4, 0:5, 0] = 4                   # a few car voxels in the pavement
    nonfree = cls[cls != FREE_CLS]
    dyn = float(np.isin(nonfree, list(range(11))).mean())
    assert dyn >= 0.010, dyn               # the old gate PASSES this
    vpc, _ = _column_stats(cls)
    assert not (vpc >= 2.0)                # the new gate does not
