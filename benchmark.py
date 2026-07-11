"""Run every detector on a video, timing throughput and (if ground truth exists) accuracy.

Usage:
  python benchmark.py data/test.mp4                 # all detectors
  python benchmark.py data/test.mp4 --only torch-cuda,psd-adaptive --tol 0.4
"""
from __future__ import annotations

import argparse
import json
import os

from detectors import REGISTRY
from evaluate import match, trap_hits


def load_gt(video_path):
    gt_path = video_path.rsplit(".", 1)[0] + ".gt.json"
    if os.path.exists(gt_path):
        return json.load(open(gt_path))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--only", default="", help="comma list of detector names")
    ap.add_argument("--tol", type=float, default=0.5, help="match tolerance seconds")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(REGISTRY)
    gt = load_gt(args.video)

    rows = []
    print(f"\nvideo: {args.video}")
    if gt:
        print(f"ground truth: {len(gt['cuts'])} cuts, {len(gt.get('traps', []))} traps, "
              f"{gt['duration_s']}s @ {gt['fps']}fps\n")

    header = f"{'detector':16} {'cuts':>5} {'fps':>8} {'xRT':>7} {'sec':>7}"
    if gt:
        header += f" {'P':>5} {'R':>5} {'F1':>5} {'traps':>6}"
    print(header)
    print("-" * len(header))

    for name in names:
        try:
            res = REGISTRY[name](args.video)
        except Exception as e:
            print(f"{name:16} ERROR: {str(e)[:60]}")
            continue
        row = res.to_dict()
        line = (f"{res.name:16} {len(res.cuts):5d} {res.analysed_fps:8.0f} "
                f"{res.realtime_x:7.0f} {res.elapsed:7.2f}")
        if gt:
            m = match(res.cuts, gt["cuts"], tol=args.tol)
            th = trap_hits(res.cuts, gt.get("traps", []), tol=args.tol)
            line += f" {m['precision']:5.2f} {m['recall']:5.2f} {m['f1']:5.2f} {len(th):6d}"
            row.update(precision=m["precision"], recall=m["recall"], f1=m["f1"],
                       tp=m["tp"], fp=m["fp"], fn=m["fn"], trap_hits=len(th))
        print(line, flush=True)
        rows.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"video": args.video, "results": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    if gt:
        print("\nP=precision R=recall F1=harmonic mean  traps=false positives on "
              "light-on/whip-pan (lower is better)")
    print("fps=frames analysed/sec  xRT=speed vs playback (e.g. 60 = 1h video in 1min)")


if __name__ == "__main__":
    main()
