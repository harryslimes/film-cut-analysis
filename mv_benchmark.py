"""Benchmark the motion-vector detector on every film that has ground truth.

For each movie under data/movies with a <stem>.gt.json it: runs the `motion-vectors`
detector (decoding once, reading the encoder's own MVs/I-frames), times it, and scores
precision/recall/F1 against the ground truth. It also scores the cached TransNetV2 cuts
(<stem>.cuts.json) side-by-side so you can see the accuracy/speed trade at a glance.

Writes <stem>.mv.srt (watchable prediction) and <stem>.mv.json per film, plus
results/mv_benchmark.json. It NEVER touches <stem>.cuts.json (the TransNet cache), so
nothing else in the pipeline is disturbed.

  python mv_benchmark.py            # all films with ground truth
  python mv_benchmark.py --tol 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from detectors import motion_vectors
from evaluate import match
from cut_times import write_srt


def gt_for(stem):
    p = stem + ".gt.json"
    if not os.path.exists(p):
        return None
    cuts = json.load(open(p)).get("cuts") or []
    return cuts or None


def discover():
    vids = []
    for ext in ("mp4", "mkv", "avi", "m4v", "webm"):
        vids += glob.glob(f"data/movies/**/*.{ext}", recursive=True)
    return sorted(set(vids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    rows = []
    for video in discover():
        stem = video.rsplit(".", 1)[0]
        gt = gt_for(stem)
        if gt is None:
            continue
        tag = os.path.basename(video)[:40]

        res = motion_vectors.detect(video)
        m = match(res.cuts, gt, tol=args.tol)

        # side-by-side: cached TransNet cuts, if present
        tn_f1 = None
        cache = stem + ".cuts.json"
        if os.path.exists(cache):
            tn = json.load(open(cache)).get("cuts") or []
            tn_f1 = match(tn, gt, tol=args.tol)["f1"]

        dur = res.n_frames / res.fps_source
        write_srt(stem + ".mv.srt", res.cuts, dur, "Shot")
        json.dump({"video": video, "detector": res.name, "fps": res.fps_source,
                   "cuts": res.cuts, "extra": res.extra},
                  open(stem + ".mv.json", "w"), indent=2)

        rows.append({
            "film": tag, "gt": len(gt), "pred": len(res.cuts),
            "P": m["precision"], "R": m["recall"], "F1": m["f1"],
            "tn_F1": tn_f1, "xRT": round(res.realtime_x),
            "fps": round(res.analysed_fps), "elapsed_s": round(res.elapsed, 1),
            "fixed_gop": res.extra.get("fixed_gop", False),
            "keyint": res.extra.get("keyint"),
        })
        flag = "  [FIXED-GOP]" if rows[-1]["fixed_gop"] else ""
        print(f"OK  {tag:42} F1={m['f1']:.3f} (P={m['precision']:.2f} "
              f"R={m['recall']:.2f})  vs TN F1={tn_f1 if tn_f1 is None else round(tn_f1,3)}"
              f"  {res.realtime_x:.0f}xRT{flag}", flush=True)

    print("\n" + "=" * 90)
    print(f"{'film':42} {'gt':>5} {'pred':>5} {'P':>5} {'R':>5} {'F1':>5} {'TN_F1':>6} {'xRT':>5}")
    print("-" * 90)
    for r in rows:
        tn = f"{r['tn_F1']:.2f}" if r["tn_F1"] is not None else "  -"
        print(f"{r['film']:42} {r['gt']:5d} {r['pred']:5d} {r['P']:5.2f} {r['R']:5.2f} "
              f"{r['F1']:5.2f} {tn:>6} {r['xRT']:5d}")
    if rows:
        import statistics
        scor = [r for r in rows if not r["fixed_gop"]]
        print("-" * 90)
        print(f"{'MEAN (scene-cut encodes only)':42} {'':5} {'':5} "
              f"{statistics.mean(r['P'] for r in scor):5.2f} "
              f"{statistics.mean(r['R'] for r in scor):5.2f} "
              f"{statistics.mean(r['F1'] for r in scor):5.2f} "
              f"{statistics.mean(r['tn_F1'] for r in scor if r['tn_F1'] is not None):6.2f} "
              f"{round(statistics.mean(r['xRT'] for r in scor)):5d}")

    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/mv_benchmark.json", "w"), indent=2)
    print(f"\nwrote results/mv_benchmark.json + per-film *.mv.srt / *.mv.json ({len(rows)} films)")


if __name__ == "__main__":
    main()
