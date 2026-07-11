"""Structural (edge-based) flash rejection for dense cut regions.

In a strobe/flash burst, raw luminance changes wildly but the scene's EDGE structure
(its 'wireframe') stays continuous; a real cut replaces the wireframe. So within
pathologically dense regions we decode the frames, measure edge discontinuity per
detected cut, and keep only the cuts with a genuine structural break -- dropping the
luminance-only flashes while PRESERVING real cuts (which pure time-collapse would lose).

Only dense regions are decoded, so cost is small. Non-dense cuts pass through untouched.
"""
from __future__ import annotations

import bisect
import subprocess

import numpy as np
import cv2


def _dense_regions(cuts, dens_window=8.0, dens_max=8, merge_gap=4.0):
    cuts = sorted(cuts)
    n = len(cuts)
    dense = [False] * n
    for i, c in enumerate(cuts):
        lo = bisect.bisect_left(cuts, c - dens_window / 2)
        hi = bisect.bisect_right(cuts, c + dens_window / 2)
        if hi - lo > dens_max:
            dense[i] = True
    regions, i = [], 0
    while i < n:
        if not dense[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (dense[j + 1] or cuts[j + 1] - cuts[j] < merge_gap):
            j += 1
        regions.append((cuts[i], cuts[j]))
        i = j + 1
    return regions


def _edge_signal(video, start, end, fps, w=256, h=144, pad=1.0):
    """Return (times, edge_discontinuity) for the region, normalised to its own median."""
    a = max(0.0, start - pad)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(a),
           "-t", str(end - a + pad), "-i", video,
           "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    F = np.frombuffer(raw, np.uint8).reshape([-1, h, w]).astype(np.float32)
    if len(F) < 3:
        return None, None
    E = np.stack([cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3),
                                cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)) for f in F])
    E = np.stack([e / (e.mean() + 1e-3) for e in E])          # normalise edge energy
    d = np.abs(E[1:] - E[:-1]).mean(axis=(1, 2))              # per-pair edge discontinuity
    d = d / (np.median(d) + 1e-6)
    times = a + (np.arange(1, len(F))) / fps
    return times, d


def refine(video, cuts, fps, edge_ratio=3.0, **region_kw):
    """Keep dense-region cuts only if they show a structural (edge) break >= edge_ratio."""
    cuts = sorted(cuts)
    regions = _dense_regions(cuts, **region_kw)
    if not regions:
        return list(cuts), {"regions": 0, "dropped": 0}
    dropped = 0
    kill = set()
    for (s, e) in regions:
        times, d = _edge_signal(video, s, e, fps)
        if times is None:
            continue
        for c in [c for c in cuts if s <= c <= e]:
            k = int(np.argmin(np.abs(times - c)))
            val = d[max(0, k - 1):k + 2].max()               # edge break near the cut
            if val < edge_ratio:                             # luminance-only flash
                kill.add(round(c, 4))
                dropped += 1
    out = [c for c in cuts if round(c, 4) not in kill]
    return out, {"regions": len(regions), "dropped": dropped}
