"""Post-process cut lists to suppress strobe/flash false positives.

Colour-flashing / strobe / animated sequences (e.g. Vertigo's Nightmare) produce bursts
of sub-second "cuts". We flag cuts sitting in a high-density neighbourhood (a strobe
region), merge them, and thin each region to a minimum spacing (`keep_gap`) -- keeping the
region's rhythm rather than deleting it, and leaving normal cutting entirely untouched.

Note: in a continuously-animated flash montage, real cuts and flashes are genuinely
indistinguishable by timing, model strength, or frame content -- even professional
references over-detect it. `keep_gap` just picks how aggressively to thin such regions.
"""
from __future__ import annotations

import bisect


def dampen_strobe(cuts, dens_window=8.0, dens_max=8, merge_gap=4.0, keep_gap=2.5):
    cuts = sorted(cuts)
    n = len(cuts)
    if n == 0:
        return []
    # 1) flag cuts whose local density is pathological (> dens_max within dens_window)
    dense = [False] * n
    for i, c in enumerate(cuts):
        lo = bisect.bisect_left(cuts, c - dens_window / 2)
        hi = bisect.bisect_right(cuts, c + dens_window / 2)
        if hi - lo > dens_max:
            dense[i] = True
    # 2) walk through; normal cuts pass; strobe cuts get thinned to keep_gap within region
    out = []
    i = 0
    last_kept = -1e9
    while i < n:
        if not dense[i]:
            out.append(cuts[i])
            i += 1
            continue
        # extend the strobe region: consecutive flagged cuts (small gaps merge across)
        j = i
        while j + 1 < n and (dense[j + 1] or cuts[j + 1] - cuts[j] < merge_gap):
            j += 1
        region_last = -1e9
        for k in range(i, j + 1):
            if cuts[k] - region_last >= keep_gap:
                out.append(cuts[k])
                region_last = cuts[k]
        i = j + 1
    return sorted(out)


def collapse_region(cuts, start, end, keep_gap=6.0):
    """Targeted override: hard-thin cuts within [start,end] seconds to keep_gap spacing.
    For a KNOWN flash/animation montage (e.g. Vertigo's Nightmare) where global rules
    can't win without harming real fast cuts elsewhere. Only touches this timespan."""
    cuts = sorted(cuts)
    out = [c for c in cuts if c < start or c > end]
    region = [c for c in cuts if start <= c <= end]
    last = -1e9
    for c in region:
        if c - last >= keep_gap:
            out.append(c)
            last = c
    return sorted(out)
