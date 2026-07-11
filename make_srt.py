"""Turn any cut list (ground-truth / synced / detected JSON) into a shot-counter SRT.

Unlike cut_times.py this does NOT run a detector -- it just renders an existing list of
cut times, so you can eyeball ground truth or a synced reference in VLC.

  python make_srt.py data/test.gt.json --out data/test.gt.srt
  python make_srt.py synced.json --video yourcopy.mp4 --out synced.srt --label Cut
"""
from __future__ import annotations

import argparse
import json

from cut_times import write_srt
from detectors.base import ffprobe_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="a json with a 'cuts' list (gt.json / cuts.json / synced)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", help="derive duration from this video if json lacks it")
    ap.add_argument("--label", default="Shot")
    args = ap.parse_args()

    data = json.load(open(args.json))
    cuts = data["cuts"]

    # duration: prefer the json, else probe the video, else just past the last cut
    if data.get("duration_s"):
        duration = float(data["duration_s"])
    elif data.get("n_frames") and data.get("fps"):
        duration = data["n_frames"] / data["fps"]
    elif args.video:
        fps, n, _, _ = ffprobe_info(args.video)
        duration = n / fps
    else:
        duration = (cuts[-1] + 5.0) if cuts else 5.0

    write_srt(args.out, cuts, duration, args.label)
    print(f"{len(cuts)} cuts -> {len(cuts)+1} shots, {duration:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
