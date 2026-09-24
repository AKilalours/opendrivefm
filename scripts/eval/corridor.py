#!/usr/bin/env python3
"""The verified-free corridor. ONE implementation, imported by every caller.

This module exists because the same quantity was computed in two files with
duplicated constants, and three separate bugs followed from that:

  A30  `safety_envelope.py` took the FIRST verified run instead of the longest,
       giving a 2.4 m median where the corridor is 95% verified to 22 m.
       `render_hd.py` already had this right and the fix was not carried across.

  A36  `render_hd.py` used clear_from = 0.4 m, which is INSIDE the road surface,
       so the displayed corridor was understated by about an order of magnitude.
       `safety_envelope.py` already had this right -- the same two files, in the
       opposite direction.

Each was caught by a number looking implausible. Neither was caught by a test.
The defect was never the constant; it was that the constant existed twice.

Constants below are the A30 values and are derived from measurement, not taste:
the height profile of non-free voxels in the lane shows levels k=0,1,2 are 100%
occupied (that is the road) and k=3 is still 54% road bleed, so the driving
envelope starts at k=4 (z >= +0.6 m) and ends at k=10 (z <= +3.4 m), below
overhead structure.
"""
from __future__ import annotations
import numpy as np

FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200

# The driving envelope. See the module docstring for the measurement.
CLEAR_FROM = 0.6      # m above ground; below this is the road surface itself
CLEAR_TO = 3.4        # m above ground; above this is overhead structure
HALF_WIDTH = 1.4      # m; a 2.8 m lane corridor
TAU = 0.15            # observability at or above which a cell counts as seen
OCC_FRAC = 0.8        # fraction of the corridor width that must be clear
BUMPER = 2.4          # m from the ego origin to the front of the vehicle
# A run that begins further ahead than this is not a corridor the ego has --
# clear road on the far side of a building is not clearance. Set from A30's
# measured near-field distribution: unverifiable near field is 1.2 m at the
# median and 12.8 m at p90, so 15 m admits the genuine near-field gap and
# rejects a patch beyond an obstacle. Found by a unit test, see A38.
MAX_START = 15.0


def verified_free(cls, obs, tau=TAU, halfw=HALF_WIDTH, clear_from=CLEAR_FROM,
                  clear_to=CLEAR_TO, occ_frac=OCC_FRAC, bumper=BUMPER,
                  max_start=MAX_START):
    """Longest run of lane ahead that is predicted free AND camera-observed.

    Returns (start_m, length_m, (j0, j1), stop_free, stop_seen) where start_m is
    how far ahead of the bumper verification begins -- never zero, because the
    cameras sit 1.5 m up and cannot see the ground at their own feet -- and
    stop_* describe what ended the run, or None if it ran to the grid edge.

    Longest run that STARTS within `max_start` of the bumper. Two failure modes
    are being avoided at once, and each was a real bug:

      first run           breaks on the marginal cell just past the bumper and
                          never restarts, giving ~2 m where the corridor runs
                          for tens of metres (A30)
      longest run anywhere credits a clear stretch on the FAR SIDE of an
                          obstacle as corridor the ego has. Clear road beyond a
                          building is not clearance (A38, found by a test)
    """
    j0 = int((RNG - halfw) / RES)
    j1 = int((RNG + halfw) / RES)
    i0 = int((RNG + bumper) / RES)
    k0 = max(int((clear_from - Z0) / RES), 0)
    k1 = min(int((clear_to - Z0) / RES), NZ)

    free = (cls[:, :, k0:k1] == FREE).all(-1)
    seen = obs.max(-1) >= tau
    ok = (free & seen)[:, j0:j1].mean(1) >= occ_frac

    i_max = i0 + int(max_start / RES)
    best_a, best_n = longest_run(ok, i0, i_max)
    if best_n == 0:
        return 0.0, 0.0, (j0, j1), None, None

    e = best_a + best_n
    sf = float(free[e, j0:j1].mean()) if e < N else None
    ss = float(seen[e, j0:j1].mean()) if e < N else None
    return (best_a - i0) * RES, best_n * RES, (j0, j1), sf, ss


def longest_run(ok, start, max_start=None):
    """Longest contiguous True run whose FIRST index is <= max_start."""
    lim = len(ok) if max_start is None else max_start
    best_a = best_n = 0
    cur = None
    for i in range(start, len(ok)):
        if ok[i]:
            if cur is None:
                cur = i if i <= lim else None
            if cur is not None and i - cur + 1 > best_n:
                best_a, best_n = cur, i - cur + 1
        else:
            cur = None
    return best_a, best_n
