"""Sync a reference cut list (MovieNet/Cinemetrics) to YOUR copy of a film.

The reference cuts live in the reference encode's timeline; your file differs by a start
offset (logos/leader) and possibly a frame-rate SCALE (23.976 vs 24 vs PAL-25 +4%). We
recover the affine map  your_time = scale * ref_time + offset  by treating each film's
cut sequence as a fingerprint and aligning them:

  1. detect cuts on YOUR file (cheap; any detector),
  2. for each plausible fps-ratio `scale`, vote for the best `offset` (Hough on pairwise
     time differences) and count matches,
  3. take the best, then ICP-refine: match -> least-squares fit scale+offset -> re-match,
     a couple of iterations. This nails scale precisely even from an approximate start.

The final match rate tells you if it's the SAME version (high) or a different cut (low).
On success it writes a ground-truth JSON REMAPPED onto your file's timeline, so the
reference labels become usable ground truth for benchmarking your detector.

  python sync.py --ref data/inception.gt.json --video inception.mkv
  python sync.py --ref data/inception.gt.json --detected inception.cuts.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from evaluate import match

# exact fps ratios worth trying (reference_fps -> your_fps), plus identity
_FPS = {"23.976": 24000 / 1001, "24": 24.0, "25": 25.0, "29.97": 30000 / 1001, "30": 30.0}
_SCALES = {1.0: "same fps"}
for a, fa in _FPS.items():
    for b, fb in _FPS.items():
        if a != b:
            _SCALES[round(fb / fa, 6)] = f"{a}->{b}"  # your_time = (fb/fa)*ref_time


def vote_offset(ref, mine, scale, tol):
    """Densest cluster of (m - scale*r) over all pairs = best offset + its support."""
    r = np.asarray(ref) * scale
    m = np.asarray(mine)
    diffs = (m[:, None] - r[None, :]).ravel()
    diffs.sort()
    win = 2 * tol
    best_n, best_b, j = 0, 0.0, 0
    for i in range(len(diffs)):
        while diffs[i] - diffs[j] > win:
            j += 1
        if i - j + 1 > best_n:
            best_n = i - j + 1
            best_b = float(diffs[(i + j) // 2])
    return best_b, best_n


def icp_refine(ref, mine, scale, offset, tol, iters=4):
    """Alternate matching and least-squares affine fit to lock in scale+offset."""
    r = np.asarray(ref)
    m = np.asarray(mine)
    for _ in range(iters):
        remapped = scale * r + offset
        res = match(list(remapped), list(m), tol=tol)  # match reference->yours
        pairs = res["matched"]
        if len(pairs) < 3:
            break
        # pairs are (predicted=remapped_ref, gt=yours); recover original ref time
        rp = np.array([(p - offset) / scale for p, _ in pairs])
        mp = np.array([g for _, g in pairs])
        A = np.vstack([rp, np.ones_like(rp)]).T
        (scale, offset), *_ = np.linalg.lstsq(A, mp, rcond=None)
    return float(scale), float(offset)


def sync(ref_cuts, my_cuts, tol=0.4):
    best = None
    for scale, label in _SCALES.items():
        off, support = vote_offset(ref_cuts, my_cuts, scale, tol)
        s2, o2 = icp_refine(ref_cuts, my_cuts, scale, off, tol)
        remapped = [s2 * r + o2 for r in ref_cuts]
        res = match(remapped, my_cuts, tol=tol)
        cand = (res["tp"], scale, s2, o2, label, res)
        if best is None or cand[0] > best[0]:
            best = cand
    tp, scale0, s, o, label, res = best
    rate = res["tp"] / max(len(ref_cuts), 1)
    return {"scale": round(s, 6), "offset": round(o, 3), "fps_hint": label,
            "matched": res["tp"], "ref_cuts": len(ref_cuts), "your_cuts": len(my_cuts),
            "match_rate": round(rate, 3), "precision": res["precision"],
            "recall": res["recall"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference cut list (gt.json or cuts.json)")
    ap.add_argument("--video", help="your copy of the film (runs a detector on it)")
    ap.add_argument("--detected", help="pre-computed cuts.json for your file")
    ap.add_argument("--detector", default="torch-cpu")
    ap.add_argument("--tol", type=float, default=0.4)
    ap.add_argument("--out", default="", help="write remapped ground truth here")
    args = ap.parse_args()

    ref = json.load(open(args.ref))["cuts"]

    if args.detected:
        my = json.load(open(args.detected))["cuts"]
        fps = json.load(open(args.detected)).get("fps", 24.0)
    elif args.video:
        from detectors import REGISTRY
        r = REGISTRY[args.detector](args.video)
        my, fps = r.cuts, r.fps_source
        print(f"detected {len(my)} cuts on your file with {r.name}")
    else:
        raise SystemExit("give --video or --detected")

    info = sync(ref, my, tol=args.tol)
    print("\n" + json.dumps(info, indent=2))
    verdict = ("SAME version — clean sync" if info["match_rate"] > 0.75 else
               "PARTIAL — probably a different cut/version, or detector missed cuts"
               if info["match_rate"] > 0.4 else
               "NO match — wrong film, wrong version, or fps not covered")
    print(f"\nverdict: {verdict}")
    print(f"map:  your_time = {info['scale']} * ref_time + {info['offset']}   ({info['fps_hint']})")

    if args.out and info["match_rate"] > 0.4:
        remapped = sorted(round(info["scale"] * c + info["offset"], 4) for c in ref)
        remapped = [c for c in remapped if c >= 0]
        json.dump({"video": args.video or "", "fps": fps, "cuts": remapped, "traps": [],
                   "source": f"synced from {args.ref}", "sync": info},
                  open(args.out, "w"), indent=2)
        print(f"wrote remapped ground truth -> {args.out}")


if __name__ == "__main__":
    main()
