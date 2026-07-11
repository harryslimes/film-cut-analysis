"""Convert a MovieNet / SceneSeg shot-annotation file into our ground-truth JSON.

These datasets give you a per-movie text file listing every shot as frame indices, e.g.

    0000 0 0 100 100
    0001 0 0 149 149
    ...
(columns vary; we take the LAST TWO integers on each line as start_frame end_frame).

A cut is the boundary between consecutive shots, i.e. each shot's start after the first.
You supply the fps (MovieNet keyframes were sampled from the source; check the movie).

  python movienet_to_gt.py shot_tt1375666.txt --fps 23.976 --out data/inception.gt.json

Then benchmark against your own copy of the movie:
  python cut_times.py inception.mkv --detector transnet-cuda --format json
  python benchmark.py inception.mkv --only transnet-cuda,psd-adaptive
"""
from __future__ import annotations

import argparse
import json
import re


def parse_shots(path):
    """MovieNet shot_detection format: `start_frame end_frame kf1 kf2 kf3` per line.
    (Some exports prefix a shot_id; we detect that and skip it.)"""
    ints_re = re.compile(r"-?\d+")
    rows = []
    for line in open(path):
        nums = [int(x) for x in ints_re.findall(line)]
        if len(nums) >= 2:
            rows.append(nums)
    if not rows:
        return []
    # 5 columns -> start,end are cols 0,1. 6 columns (leading shot_id) -> cols 1,2.
    off = 1 if all(len(r) >= 6 for r in rows[:10]) else 0
    shots = [(r[off], r[off + 1]) for r in rows]
    shots.sort()
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shotfile")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default="", help="path to your copy of the movie (optional)")
    args = ap.parse_args()

    shots = parse_shots(args.shotfile)
    if not shots:
        raise SystemExit("no shots parsed -- check the file format")

    # cut = first frame of every shot after the first
    cut_times = sorted(round(s[0] / args.fps, 4) for s in shots[1:])
    n_frames = shots[-1][1] + 1

    gt = {
        "video": args.video, "fps": args.fps, "n_frames": n_frames,
        "duration_s": round(n_frames / args.fps, 3),
        "source": f"MovieNet/SceneSeg shot annotation: {args.shotfile}",
        "cuts": cut_times, "traps": [],
    }
    json.dump(gt, open(args.out, "w"), indent=2)
    print(f"{len(shots)} shots -> {len(cut_times)} cuts, {gt['duration_s']}s @ {args.fps}fps")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
