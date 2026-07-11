"""Extract cut times from a video and write them as CSV / EDL / JSON.

  python cut_times.py movie.mkv                         # -> movie.cuts.csv
  python cut_times.py movie.mkv --detector transnet-cuda --format edl
  python cut_times.py movie.mkv --detector psd-adaptive --format all

Pick a detector by name (see `python cut_times.py --list`). Sensible default is a fast,
brightness/pan-robust one; use transnet-* for best accuracy on real footage.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from detectors import REGISTRY


def hhmmssff(seconds, fps):
    f = int(round(seconds * fps))
    ff = f % int(round(fps))
    s = f // int(round(fps))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}:{ff:02d}"


def write_csv(path, cuts, fps):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["cut_index", "time_seconds", "timecode"])
        for i, c in enumerate(cuts, 1):
            wr.writerow([i, f"{c:.3f}", hhmmssff(c, fps)])


def srt_ts(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path, cuts, duration, label="Shot"):
    """One subtitle per shot; the number increments at each cut (watch in VLC)."""
    bounds = [0.0] + list(cuts) + [duration]
    with open(path, "w") as fh:
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            if end <= start:
                continue
            fh.write(f"{i+1}\n{srt_ts(start)} --> {srt_ts(end)}\n"
                     f"{label} {i+1}\n\n")


def write_edl(path, cuts, fps, duration):
    # minimal CMX3600-style EDL: one clip per shot
    bounds = [0.0] + list(cuts) + [duration]
    with open(path, "w") as fh:
        fh.write("TITLE: cut-time-extractor\nFCM: NON-DROP FRAME\n\n")
        for i in range(len(bounds) - 1):
            src_in = hhmmssff(bounds[i], fps)
            src_out = hhmmssff(bounds[i + 1], fps)
            fh.write(f"{i+1:03d}  AX       V     C        "
                     f"{src_in} {src_out} {src_in} {src_out}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--detector", default="torch-cpu")
    ap.add_argument("--format", default="csv", choices=["csv", "edl", "srt", "json", "all"])
    ap.add_argument("--label", default="Shot", help="subtitle label prefix (SRT)")
    ap.add_argument("--list", action="store_true", help="list available detectors")
    args = ap.parse_args()

    if args.list or not args.video:
        print("available detectors:")
        for k in REGISTRY:
            print("  ", k)
        return

    if args.detector not in REGISTRY:
        raise SystemExit(f"unknown detector '{args.detector}'. try --list")

    res = REGISTRY[args.detector](args.video)
    stem = args.video.rsplit(".", 1)[0]
    duration = res.n_frames / res.fps_source

    print(f"{res.name}: {len(res.cuts)} cuts in {res.elapsed:.1f}s "
          f"({res.analysed_fps:.0f} fps, {res.realtime_x:.0f}x realtime)")

    wrote = []
    if args.format in ("csv", "all"):
        p = stem + ".cuts.csv"; write_csv(p, res.cuts, res.fps_source); wrote.append(p)
    if args.format in ("edl", "all"):
        p = stem + ".cuts.edl"; write_edl(p, res.cuts, res.fps_source, duration); wrote.append(p)
    if args.format in ("srt", "all"):
        p = stem + ".cuts.srt"; write_srt(p, res.cuts, duration, args.label); wrote.append(p)
    if args.format in ("json", "all"):
        p = stem + ".cuts.json"
        json.dump({"video": args.video, "detector": res.name, "fps": res.fps_source,
                   "cuts": res.cuts}, open(p, "w"), indent=2); wrote.append(p)
    for p in wrote:
        print("wrote", p)


if __name__ == "__main__":
    main()
