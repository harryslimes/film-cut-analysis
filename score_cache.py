"""Batch score-cache format discrimination (CineScript S3-F1, amendment A3.3).

Stdlib only. batch_score.py caches its per-film TransNet results as <stem>.cuts.json.
A reader of that cache must accept ONLY its own tagged score-cache, or a narrow
untagged legacy {cuts, fps} cache -- and REJECT a v2 cut-events document (which also
has `cuts`/`fps` at top level) or any other/unknown format, so a corpus is never
silently poisoned by reading the wrong kind of file as ground-truth input.
"""
from __future__ import annotations

import math

SCORE_CACHE_FORMAT = "film-cut-analysis/score-cache"


def load_score_cache(d):
    """Return (cuts, fps) from a batch score-cache dict, or raise ValueError.

    Accepts:
      - the tagged score-cache (``format == SCORE_CACHE_FORMAT``), or
      - a narrow untagged legacy cache: no ``format`` key, keys a subset of
        {cuts, fps}, with a ``cuts`` list present.
    Both forms require BOTH a ``cuts`` list AND a valid ``fps`` (a positive finite
    number, not bool) -- a missing fps is rejected, never invented (S3-F2 gap 3).
    Rejects v2 cut-events documents and every other / unknown format."""
    if not isinstance(d, dict):
        raise ValueError("cache is not a JSON object")
    fmt = d.get("format")
    if fmt == SCORE_CACHE_FORMAT:
        pass
    elif fmt is None and "cuts" in d and set(d) <= {"cuts", "fps"}:
        pass
    else:
        raise ValueError(f"unrecognised cache format {fmt!r}")
    cuts = d.get("cuts")
    if not isinstance(cuts, list):
        raise ValueError("cache has no cuts list")
    fps = d.get("fps")
    if not (isinstance(fps, (int, float)) and not isinstance(fps, bool) and math.isfinite(fps) and fps > 0):
        raise ValueError(f"cache fps missing or invalid ({fps!r})")
    return cuts, fps
